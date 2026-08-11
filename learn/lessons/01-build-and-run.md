# 01 — Build a model by hand and run it

## Read

- `doc/graph.md` §"Graph, Node, and OutletId" — the struct definitions. The one
  idea to hold on to: an `OutletId` is `(node, slot)` and *is* the wire.
- `core/src/lib.rs` — the crate doc example is a five-line model. This lesson's
  model is that one with three more nodes.

## The model

`learn/src/lib.rs` builds this:

```
input [16, 8] ──── mul(input, ones) ──── add(mul, addc) ──── matmul(add, weights) ──► [16, 4]
ones  [1, 8]  ────┘                     │
bias_a [16, 8] ─┬── addc(bias_a, bias_b)┘
bias_b [16, 8] ─┘
```

Every node is there for a reason that only becomes visible in Lesson 02:

| Node | Why it exists |
|---|---|
| `mul` | multiplies by 1 — a no-op, for a pass to delete |
| `addc` | two constant inputs — for constant folding to eat |
| `add`, `matmul` | real work; these carry the numerics |
| `matmul` | an `EinSum`, for codegen to lower |

## Predict

1. `M = 16`, `K = 8`, `N = 4`. What are the input and output facts?
2. `ones` has shape `[1, 8]`, not `[8]`. Why might a rank-1 constant be a
   problem when the other operand is `[16, 8]`?
3. The output rows: `input[r, c] = (r * 8 + c) / 10`, times 1, plus `0.25 + 0.75`,
   then matmul with an all-`0.5` weight matrix. What is `output[0, 0]`?

## Run

```sh
cargo run -p tract-learn --bin lesson01
```

```
<the graph dump — decoded below>

input  fact: 16,8,F32
output fact: 16,4,F32

input:  16,8,F32 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1...
output: 16,4,F32 5.4, 5.4, 5.4, 5.4, 8.6, 8.6, 8.6, 8.6, 11.8, 11.8, 11.8, 11.8...

row 0 by hand: sum over k of (input[0,k] + 1) * 0.5 = 5.4
```

All four columns of a row are equal because every weight column is identical —
deliberate, so a wrong answer is obvious at a glance.

## Explain

### Reading the dump

`println!("{model}")` goes through `impl Display for Graph` in
`core/src/model/graph.rs`. One line per node, from one format string:

```
"{:5} | {:8} {:8} -> {:8} {:8} | {:25} {:50} {} => {}"
  id     in#0     in#1        succ     succ    op name  node name  input facts => output facts
```

Column widths are fixed, so unused slots come out as runs of spaces.

**The two index notations are different types.** This is the whole trick. Their
`Debug` impls (`core/src/model/node.rs`) differ only in where the `>` sits:

| Printed | Type | Reads as |
|---|---|---|
| `2/0>` | `OutletId { node: 2, slot: 0 }` | data flowing **out of** node 2's output slot 0 |
| `>6/1` | `InletId { node: 6, slot: 1 }` | data flowing **into** node 6's input slot 1 |

The `>` is an arrowhead pointing away from the node. So left of `->` are this
node's inputs as `OutletId`s — where my data comes from; right of `->` are its
consumers as `InletId`s — where my data goes, and into which argument position.

`slot` means something different on each side. On an outlet it is *which output*
of that node — almost always `0`, most ops have one. On an inlet it is *which
argument* of the consumer: `>6/0` is node 6's first argument, `>6/1` its second.

The lesson model, trimmed to the columns that matter:

```
 id | inputs      -> consumers | op      name      facts
  0 |             -> >2/0      | Source  input     => 16,8,F32
  1 |             -> >2/1      | Const   ones      => 1,8,F32 …
  2 | 0/0>  1/0>  -> >6/0      | Mul     mul       16,8,F32 ; 1,8,F32 … => 16,8,F32
  3 |             -> >5/0      | Const   bias_a
  4 |             -> >5/1      | Const   bias_b
  5 | 3/0>  4/0>  -> >6/1      | Add     addc
  6 | 2/0>  5/0>  -> >8/0      | Add     add
  7 |             -> >8/1      | Const   weights
  8 | 6/0>  7/0>  ->           | EinSum  matmul    => 16,4,F32
```

Node 2 reads: my arg 0 is node 0's output (`input`), my arg 1 is node 1's output
(`ones`), and my result is consumed as arg 0 of node 6. Node 8's consumer column
is empty because nothing consumes it — it is the graph output, which the trailing
`outputs: 8/0>` line states (in `OutletId` form again).

Map that back onto `learn/src/lib.rs`: each `wire_node("name", op, &[a, b])` is
one node whose `inputs` vec is exactly `[a, b]`. The positions in that `&[..]`
*are* the inlet slots.

**Every edge is printed twice, on purpose.** Once as an `OutletId` in the
consumer's row, once as an `InletId` in the producer's row. `Node.inputs:
Vec<OutletId>` and `Outlet.successors: Vec<InletId>` are two stored views of the
same edge set, kept in sync so the graph walks cheaply in both directions —
patches and rewrite rules need the successor direction to know who to re-wire.
Rows that disagree mean a corrupt graph.

**Two truncation traps.** The line shows at most two inputs, and at most two
consumers *of output slot 0*. Past that, continuation lines appear:

- `* inputs: …` — the full list, when a node has more than 2 inputs.
- `* output #N: <label> <inlets>` — per-slot successors, when the node has
  several outputs, more than 2 successors, or a labelled outlet.

This model triggers neither, so every node fits on one line. Do not read an empty
consumer column as "no consumers" without checking for those lines.

**The facts half** is `input facts => output facts`, slots joined by ` ; `. Const
rows have nothing left of `=>`: no inputs. The emoji come from `impl Debug for
TypedFact` (`core/src/model/fact.rs`):

- `🟰 <tensor>` — the fact carries a known constant value (`konst`). This is what
  makes a node foldable.
- `◻️ ,F32 1` — the fact is *uniform*: every element is the same scalar. Cheap to
  test, and several passes special-case it.

So `1,8,F32🟰 1,8,F32 1, 1, … ◻️ ,F32 1` is a `1×8` f32 whose value is known and
all-ones.

One last thing, from the doc comment on `Node::id`: **node ids are not stable
across transformations.** After Lesson 02's declutter the same op may sit at a
different id. Never hard-code one; match on names or ops.

### The five calls that build any model

```rust
let mut model = TypedModel::default();
let input = model.add_source("input", f32::fact([M, K]))?;      // -> OutletId
let ones  = model.add_const("ones", tensor)?;                    // -> OutletId
let mul   = model.wire_node("mul", mul(), &[input, ones])?[0];   // -> TVec<OutletId>
model.select_output_outlets(&[matmul])?;
```

- `wire_node` returns a `TVec<OutletId>`, one per output slot. Most ops have one,
  hence the `[0]` everywhere.
- The method is **`select_output_outlets`**. There is no `set_output_outlets`;
  `auto_outputs()` also exists and guesses from the topology (any node with no
  successors), which is what the `core/src/lib.rs` example uses. Prefer being
  explicit.
- `f32::fact([M, K])` builds a `TypedFact`: element type plus shape. Facts are
  the thing every pass reasons about.

### Typed ops do not broadcast ranks

This was prediction 2. Try making `ones` rank 1 and you get a build-time error,
not a silent broadcast:

```
in output_facts invocation for mul: Mul
Caused by: Typed ops require rank match. Invalid inputs for Mul: 16,8,F32 ; 8,F32
```

`TypedBinOp` requires **equal rank**; it broadcasts *dimensions* (`1` against
`n`), never ranks. NumPy and ONNX both promote ranks implicitly, so this is a
real trap when hand-building. Framework importers insert the missing `AddAxis`
node themselves — which is one reason an imported graph has more nodes than the
source file suggests.

### Running

```rust
let plan = SimplePlan::new(model.clone())?;
let outputs = plan.run(tvec!(input.into()))?;
```

`SimplePlan::new` computes an evaluation order and allocates. Note what it does
*not* do: **it does not optimise.** This model runs, and gives numerically
correct answers, as the raw graph you just wired. `into_optimized()` is a
separate call — Lesson 02.

That separation is the single most useful fact in this course, and the source of
tract's classic "why is it slow" report: a decluttered-but-unoptimised graph is
correct and several times slower than what you would ship.

## Exercise

Add a second output to the model — `select_output_outlets(&[add_node, matmul])`
— and re-run. What changes in the dump, and what does `plan.run` return now?
Then remove `mul` from the chain entirely and confirm the numerics are unchanged;
you have just done by hand what Lesson 02 watches a pass do.

---

Next: [02 — The stages, one at a time](02-stages.md)
