# 02 — The stages, one at a time

## Read

- `doc/pipeline.md` §"Decluttered vs optimised" — two paragraphs, the crux of
  the whole course.
- `doc/pipeline.md` §"Example rewrites" — note that declutter goes in *both*
  directions: it decomposes some ops and fuses others.

## Predict

The model from Lesson 01 has 9 nodes: `{Add: 2, Const: 4, EinSum: 1, Mul: 1,
Source: 1}`.

1. After `declutter()`, which nodes are gone? How many are left?
2. After `optimize()`, what happens to `Add` and to `EinSum`?
3. `weights` is a `Const` feeding the matmul. Is it still a `Const` node after
   `optimize()`?
4. Do all three stages produce the same numbers?

## Run

```sh
cargo run -p tract-learn --bin lesson02
```

### After wiring — 9 nodes

```
op histogram: {"Add": 2, "Const": 4, "EinSum": 1, "Mul": 1, "Source": 1}
```

### After `declutter()` — 5 nodes

```
0 | Source        input     => 16,8,F32
1 | Const         addc      => 16,8,F32  1, 1, 1, 1, ...
2 | Add           add       => 16,8,F32
3 | Const         weights   => 8,4,F32   0.5, 0.5, ...
4 | EinSum        matmul    => 16,4,F32

op histogram: {"Add": 1, "Const": 2, "EinSum": 1, "Source": 1}
```

Two things happened. `mul` is **gone** — not replaced, deleted, and its input
wired straight through to `add`. And `addc` is no longer an `Add`: it is now a
single `Const` holding `1.0` (that is `0.25 + 0.75`), with `bias_a` and `bias_b`
removed. Four nodes became one.

### After `optimize()` — still 5 nodes, but different ones

```
0 | Source         input          => 16,8,F32
1 | Const          addc           => 16,8,F32
2 | OptAddUnicast  add            => 16,8,F32
3 | OptMatMulPack  matmul.pack_a  => ,F32 🔍 DynPackedExoticFact { k: Val(8), mn: Val(16), packers: [PackedF32[8]@16+1] }
4 | OptMatMul      matmul         => 16,4,F32

op histogram: {"Const": 1, "OptAddUnicast": 1, "OptMatMul": 1, "OptMatMulPack": 1, "Source": 1}
```

The node *count* barely moved; the node *identities* changed completely. And:

```
declutter == pre-declutter: true
optimize  == pre-declutter: true
```

## Explain

### Declutter deleted `mul`

`declutter_neutral` in `core/src/ops/binary.rs`. It fires when one input is
*uniform* (all elements equal) and that value is the op's `neutral_element()` —
`1` for `Mul`, `0` for `Add`. `Mul` is commutative, so operand order doesn't
matter. The patch is a `rewire`: the node vanishes and its variable input is
shunted to its consumers.

Note "uniform", not "scalar". A `[1, 8]` of all ones qualifies, and so would a
`[16, 8]` of all ones.

### Declutter folded `addc`

`PropConst` (`core/src/optim/prop_const.rs`), the first pass in
`Optimizer::declutter()`. Any stateless node whose inputs are all constant gets
evaluated at compile time and replaced by a `Const`.

### `optimize()` lowered both real ops

- `Add` → **`OptAddUnicast`**, a linalg-backed elementwise kernel. Codegen
  rewrites elementwise ops too, not just the matmul-shaped ones — worth knowing
  before you go hunting for a plain `Add` in an optimised dump and conclude
  something broke.
- `EinSum` → **`OptMatMulPack`** + **`OptMatMul`**. The packing node rearranges
  the activations into the layout the micro-kernel wants. Its output fact has an
  *empty shape* and an exotic fact (`DynPackedExoticFact`) — after codegen,
  values in the graph are no longer plain row-major arrays, which is exactly why
  this form is not portable.
- `weights` **disappeared as a node** (prediction 3: `Const` went 2 → 1). The
  constant was packed at compile time and baked into the `OptMatMul` op itself.
  Lesson 04 catches this happening.

`OptMatMul` is the current name. Older material calls it `LirMatMul`; that type
no longer exists in this tree.

## The trap: two different constant folders

Run the second half of the output — the same model with `BiasVolume::BelowEagerFoldLimit`,
where the biases are `[1, 8]` (volume 8) instead of `[16, 8]` (volume 128).

Its *pre-declutter* graph already has only 8 nodes and **one** `Add`:

```
1 |  -> >2/1  >5/1 | Const  ones    => 1,8,F32  1, 1, ...
3 |  ->            | Const  bias_a  => 1,8,F32  0.25, ...
4 |  ->            | Const  bias_b  => 1,8,F32  0.75, ...
5 | 2/0>  1/0>     | Add    add
```

There is no `addc` node at all, and `bias_a`/`bias_b` are dangling — no
successors. Declutter has not run yet. Three mechanisms fired during *wiring*:

1. **`wire_node` const-folds eagerly.** `core/src/model/typed.rs` evaluates a
   stateless op at build time when every input is a constant with
   `volume() < 16 && is_plain()`. Volume 8 qualifies; volume 128 does not.
2. **`wire_node` deduplicates `Const` ops by value**, before anything else. The
   folded result was `[1, 8]` of all ones — identical to the existing `ones`
   node, so it returned *that* outlet. Look at node 5: its second input is
   `1/0`, the `ones` node. The name `addc` was silently discarded.
3. `bias_a` and `bias_b` are left dangling and survive until declutter compacts
   the graph.

Both variants converge on the same 5-node decluttered graph, so nothing is
*wrong* here. But if you are trying to observe a declutter rule and your fixture
uses small constants, **the effect you are looking for may already have happened
before your first dump** — and you will spend an afternoon concluding the pass is
broken. Size your fixtures above volume 16, or dump immediately after each
`wire_node`.

This is not hypothetical. `harness/nnef-test-cases/sign-abs-integers/graph.nnef`
folds its constant legs, and the comment there credits declutter — but its
constants are volume 6, so the eager path in `wire_node` gets there first.

## Exercise

`into_optimized()` is `declutter()` then `optimize()`. Call `optimize()` on a
model you have **not** decluttered and diff the result against the proper
pipeline. Do you get the same graph? Look at `Optimizer::codegen()` in
`core/src/optim/mod.rs` before you guess — the answer is in its pass list, and it
is the seed of Lesson 04.

<details>
<summary>Answer</summary>

Identical graph, identical numerics. `Optimizer::codegen()` contains both a
`PropConst` pass and an `OpOptim("declutter", …)` pass, so the declutter rules
run *again* during optimize. Skipping `declutter()` costs you nothing on a model
this small.

Do not read that as "declutter is optional". It is the stage boundary that
matters: decluttered is the portable form you serialise and ship, and you cannot
get back to it from an optimised graph.

</details>

---

Next: [03 — The same model through the CLI](03-same-model-via-cli.md)
