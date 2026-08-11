# 04 — Reading the pass lists

## Read

- `core/src/optim/mod.rs` — `Optimizer::declutter()` and `Optimizer::codegen()`.
  Both are literal `vec![]`s of passes. Read the two lists; that is the whole
  configuration of the two stages you have been watching.
- `AGENTS.md` §"Model rewriting" — the "when to use which" table.

## The two lists

```rust
pub fn declutter() -> Optimizer {
    Optimizer::passes(vec![
        Box::<PropConst>::default(),
        Box::<PropagateUniformTdim>::default(),
        Box::<PropagateRoi>::default(),
        Box::<FoldUniformMask>::default(),
        Box::new(OpOptim("declutter", TypedOp::declutter_with_session, 0)),
        Box::new(PushSliceUp),
        Box::new(PushSplitDown),
        Box::<concat_then_einsum::ConcatThenEinsum>::default(),
        Box::<ChangeAxes>::default(),
    ])
}

pub fn codegen() -> Optimizer {
    Optimizer::passes(vec![
        Box::<PropConst>::default(),
        Box::<MergeConsecutiveSameRoleAxes>::default(),
        Box::new(OpOptim("codegen", TypedOp::codegen, 0)),
        Box::new(OpOptim("declutter", TypedOp::declutter_with_session, 0)),
        Box::new(PushSplitDown),
        Box::new(OpOptim("fuse", TypedOp::fuse, 0)),
    ])
}
```

Note what `codegen()` contains: `PropConst` **and** the declutter pass. Optimise
re-runs decluttering. That is why the Lesson 02 exercise found that skipping
`declutter()` changed nothing.

Note also that most rules do not live in these lists at all. `OpOptim("declutter",
…)` just walks nodes calling `TypedOp::declutter` on each op. The rules are
methods on the ops — `declutter_neutral` is in `core/src/ops/binary.rs`, next to
`Mul`, not in a central registry. `Op::declutter` and `Op::codegen` are the two
hooks; `doc/op.md` covers the trait.

## Predict

The optimizer applies one patch at a time and counts them. `Optimizer::stopping_at(n)`
stops after `n` patches — the library form of the CLI's `--declutter-step N`.

Starting from the 9-node model: which change happens on patch **1**, the `Mul`
deletion or the `addc` fold? Use the pass order to decide.

## Run

```sh
cargo run -p tract-learn --bin lesson04
```

```
declutter, one patch at a time:
  stopping_at(0) -> {"Add": 2, "Const": 4, "EinSum": 1, "Mul": 1, "Source": 1}
  stopping_at(1) -> {"Add": 1, "Const": 3, "EinSum": 1, "Mul": 1, "Source": 1}
  stopping_at(2) -> {"Add": 1, "Const": 2, "EinSum": 1, "Source": 1}
  stopping_at(3) -> {"Add": 1, "Const": 2, "EinSum": 1, "Source": 1}
  stopping_at(4) -> {"Add": 1, "Const": 2, "EinSum": 1, "Source": 1}
```

Patch 1 folds `addc` — `PropConst` is first in the list, and it gets its turn
before the per-op declutter pass ever runs. Patch 2 deletes the `Mul`. By patch 3
the graph is at its fixpoint and further patches are no-ops.

```
and the same for optimize()'s codegen list:
  stopping_at(0) -> {"Add": 1, "Const": 2, "EinSum": 1, "Source": 1}
  stopping_at(1) -> {"Const": 2, "EinSum": 1, "OptAddUnicast": 1, "Source": 1}
  stopping_at(2) -> {"Const": 2, "EinSumMatMul": 1, "OptAddUnicast": 1, "Source": 1}
  stopping_at(3) -> {"Const": 2, "OptAddUnicast": 1, "OptMatMul": 1, "OptMatMulPack": 2, "Source": 1}
  stopping_at(4) -> {"Const": 2, "OptAddUnicast": 1, "OptMatMul": 1, "OptMatMulPack": 1, "Source": 1}
```

This is the interesting one. Read it as a filmstrip:

1. `Add` → `OptAddUnicast`.
2. `EinSum` → **`EinSumMatMul`**. An op you have never seen in a finished dump,
   because it only exists *between two codegen patches*. `EinSum::codegen`
   recognises the matmul shape and rewrites to this intermediate op, which then
   has its own `codegen` that picks the kernel.
3. `EinSumMatMul` → `OptMatMul` plus **two** `OptMatMulPack` nodes — one packing
   the activations, one packing the constant weights.
4. `OptMatMulPack` drops 2 → 1. The weights-packing node had constant input, so
   it got folded and then **baked into the `OptMatMul` op itself**
   (`bake_const_operands` in `core/src/ops/matmul/optimized.rs`). That is where
   the `weights` `Const` node from Lesson 02 went.

Weight packing happening at compile time, once, is a large part of why the
optimised form is fast — and why it is machine-specific and cannot be
serialised.

## The CLI equivalent

`--declutter-step N` does the same bisection on the CLI:

```sh
./target/debug/tract learn/models/tiny --declutter-step 1 dump
```

Our checked-in model is already decluttered, so this shows nothing on it; the
flag earns its keep on an ONNX model, where you use it to find *which* patch broke
something. Turn on `-v` to see the stage log:

```sh
./target/debug/tract -v learn/models/tiny --pass declutter dump -q
```

`--optimize-step N` looks like the codegen counterpart. It is declared and never
read — see Lesson 08.

## Which tool for which job

From `AGENTS.md`, the table worth memorising:

| Situation | Tool |
|---|---|
| Simplify or fuse one op type | `Op::declutter` + `TypedModelPatch` |
| Cross-op pattern (N ops → M ops) | `Rewriter` rule |
| Whole-model structural change | `ModelTransform` |
| Backend lowering for one op | `Op::codegen` + `TypedModelPatch` |

And the rule underneath it: **never hand-roll a model-walk loop.** Build a patch
and let the driver apply it. Lesson 05 does exactly that.

## Exercise

Pick one pass from `Optimizer::declutter()` that this model never exercises —
`PushSliceUp`, say — open its file, and work out what fixture *would* trigger it.
Then build that fixture in `learn/src/lib.rs` and confirm with `stopping_at`.

---

Next: [05 — Writing your own rule](05-your-own-rule.md)
