# 05 — Writing your own rule

## Read

- `AGENTS.md` §"Model rewriting" — `TypedModelPatch`, `Rewriter`, and the three
  guard macros.
- `core/src/model/rewriter.rs` — 60 lines. Read `with_rule_for` and `rewrite`.
- A real rule for shape: `declutter_reduce_reduce` in `core/src/ops/nn/reduce.rs`
  merges two consecutive reductions. It is the closest existing rule to what you
  are about to write.

## The target

`two_adds_model()` in `learn/src/lib.rs`:

```
input ── add_c1(input, c1) ── add_c2(_, c2) ──► output
```

Two additions of a constant, back to back. Neither is constant-foldable on its
own: each has the running value as one input, so `PropConst` cannot touch either.

```sh
cargo run -p tract-learn --bin lesson05
```

```
=== after stock declutter() — does tract already fold this? ===
op histogram: {"Add": 2, "Const": 2, "Source": 1}
nodes: 5
```

Confirmed: tract does not fold this today. There is real work to do.

## Predict

You want `Add(Add(x, c1), c2)` to become one addition. Two ways to write it:

- **(a)** compute `c1 + c2` yourself in the rule and emit `Add(x, sum)`.
- **(b)** re-associate to `Add(x, Add(c1, c2))` and let an existing pass fold the
  constant-only node.

Which is less code, and which is more in keeping with tract's design? What has to
be true about the *inner* `Add` for either to be a win?

## The rule

`learn/src/lib.rs` takes option **(b)**:

```rust
fn hoist_constant_adds(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    op: &TypedBinOp,
) -> TractResult<Option<TypedModelPatch>> {
    rule_if!(op.0.is::<Add>());
    rule_if_some!((outer_var, outer_const) = split_const(model, node));

    let inner = model.node(outer_var.node);
    rule_if_some!(inner_op = inner.op_as::<TypedBinOp>());
    rule_if!(inner_op.0.is::<Add>());
    rule_if!(inner.outputs[0].successors.len() == 1);
    rule_if!(!model.outputs.contains(&outer_var));
    rule_if_some!((x, inner_const) = split_const(model, inner));

    let mut patch = TypedModelPatch::default();
    let taps = patch.taps(model, &[x, inner_const, outer_const])?;
    let folded = patch.wire_node(format!("{name}.consts"), add(), &[taps[1], taps[2]])?[0];
    let out = patch.wire_node(format!("{name}.hoisted"), add(), &[taps[0], folded])?[0];
    patch.shunt_outside(model, node.id.into(), out)?;
    Ok(Some(patch))
}
```

Four things to take from this shape:

- **The guards read as preconditions.** `rule_if!` bails when false,
  `rule_if_some!` bails on `None`, both returning `Ok(None)` — "this rule does
  not apply here", which is not an error. Defined in `core/src/transform.rs`.
- **`op.0.is::<Add>()`** is how you test a binary op's identity. `TypedBinOp` is
  a wrapper around a `Box<dyn BinMiniOp>`; the mini-op carries the arithmetic.
  Same idiom as `core/src/ops/nn/rms_norm.rs`.
- **`patch.taps`** creates the patch's handles on existing wires. You build the
  replacement subgraph inside the patch, then `shunt_outside` redirects the old
  node's consumers to your new outlet. You never mutate the model.
- **Both liveness guards are load-bearing.** The patch only shunts the outer
  node; the inner `Add` disappears solely by becoming dead and being dropped by
  `compact()`. If anything still holds it, hoisting duplicates the addition
  instead of removing it, and a rule that ignores this makes graphs *slower*.
  Two distinct things can hold it, hence two guards: another consumer node, and
  the model's own output list — `successors` counts consumers only, so
  `model.outputs.contains` is a separate check. Same idiom as
  `core/src/optim/concat_then_einsum.rs`.

Wrapped as a `ModelTransform`:

```rust
impl ModelTransform for HoistConstantAdds {
    fn name(&self) -> StaticName { "hoist-constant-adds".into() }
    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        Rewriter::default()
            .with_rule_for::<TypedBinOp>("hoist-constant-adds", hoist_constant_adds)
            .rewrite(&(), model)
    }
}
```

## The result

```
=== after HoistConstantAdds ===
0 | Source  input
1 | Const   add_c2.consts   => 16,8,F32  1, 1, 1, 1, ...
2 | Add     add_c2.hoisted  => 16,8,F32

op histogram: {"Add": 1, "Const": 1, "Source": 1}
nodes: 3

numerics unchanged by the rule: true
```

Two `Add`s became one, and two constants became one — note `add_c2.consts` is
already a `Const` holding `1.0`, not an `Add`. **You never wrote the arithmetic.**
`Rewriter::rewrite` calls `model.prop_consts()` after each round, so the
constant-only `Add` you created was folded for you.

That is the design principle from `AGENTS.md` in miniature: *model-wide behaviour
emerges from op-scoped manipulations composing together.* Your rule only had to
know how to move a wire.

## Termination

A rewrite rule that can re-fire on its own output loops forever. Check yours:
after the patch, the outer `Add`'s variable input is `x`, whose producer is a
`Source`, not an `Add`. The guard fails and the rule stops. `Rewriter::rewrite`
loops to a fixpoint, so this matters.

## Exercise

Two, in increasing difficulty:

1. **Break the guards.** Comment out one guard line, run
   `cargo test -p tract-learn`, restore it, then do the same with the other. Each
   guard has exactly one test that catches it: `successors.len() == 1` →
   `lesson05_rule_leaves_shared_inner_add_alone`, `model.outputs.contains` →
   `lesson05_rule_leaves_inner_add_that_is_an_output_alone`. Read both fixtures
   to see why the rule must decline, and note what the tests assert: **node
   names, not the op histogram.** On these fixtures a wrong fire swaps one
   `Const` and one `Add` for one of each, so the op counts stay identical and the
   numerics stay correct — a histogram assertion would pass on a broken rule.
   Picking an oracle that can actually observe the damage is half of testing a
   rewrite.
2. **Generalise it.** Make the rule work for `Mul` as well as `Add`. You will
   need the mini-op to be the *same* on both nodes — and think about whether
   `Add(Mul(x, c1), c2)` is safe to touch (it isn't; why?).

Then, if you want the real thing: this rule lives in `learn/` because it is a
course exercise. A version of it belonging in `core` would go in
`core/src/ops/binary.rs` next to `declutter_neutral`, as an `Op::declutter`
method rather than a standalone transform, with a test in the relevant `suite-*`
crate. That is the shape of a genuine PR.

---

Next: [06 — Symbolic shapes](06-symbolic-shapes.md)
