//! Course fixtures for the tract compilation pipeline.
//!
//! Everything here builds on the internal crates (`tract-core`, `tract-nnef`)
//! rather than the public `api/rs` surface. That is deliberate and specific to
//! this course: `api/rs` collapses the pipeline stages (`NnefInterface::load`
//! already declutters, `into_runnable` hides optimize behind the plan), so it
//! cannot show an intermediate graph. Real client code should still use
//! `api/rs` only.

use std::collections::BTreeMap;

use tract_core::internal::*;
use tract_core::model::Rewriter;
use tract_core::ops::binary::TypedBinOp;
use tract_core::ops::einsum::EinSum;
use tract_core::ops::math::{Add, add, mul};
use tract_core::transform::ModelTransform;

/// Volume of the two constants feeding the foldable `add` in [`tiny_model`].
///
/// `TypedModel::wire_node` const-folds a stateless op at build time when every
/// input is a constant of `volume() < 16`. So this knob decides *which* of the
/// two constant-folding mechanisms gets to the `bias_a + bias_b` node first:
/// the eager path inside `wire_node`, or the `PropConst` pass during declutter.
///
/// Both variants are rank 2: `TypedOp` binaries require their inputs to have
/// equal rank, so a rank-1 `[K]` constant against a `[M, K]` input is rejected
/// by `output_facts` rather than implicitly broadcast. Framework importers
/// insert the missing `AddAxis` themselves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BiasVolume {
    /// `[1, 8]` constants, volume 8. Folded eagerly by `wire_node`, so the node
    /// is already gone before declutter ever runs.
    BelowEagerFoldLimit,
    /// `[16, 8]` constants, volume 128. Survives wiring and is folded by
    /// `PropConst` during declutter.
    AboveEagerFoldLimit,
}

impl BiasVolume {
    fn shape(&self) -> &'static [usize] {
        match self {
            BiasVolume::BelowEagerFoldLimit => &[1, K],
            BiasVolume::AboveEagerFoldLimit => &[M, K],
        }
    }
}

/// Rows of the input, and of the matmul output. Kept `>=` [`N`] so `EinSum`
/// codegen does not transpose the operands before lowering.
pub const M: usize = 16;
/// Shared dimension: input columns and weight rows.
pub const K: usize = 8;
/// Weight columns, and matmul output columns.
pub const N: usize = 4;

/// A four-node model where every node exists to be acted on by one named pass.
///
/// ```text
/// input  [16, 8] ─┬─ mul(input, ones) ─── add(mul, addc) ── matmul(add, weights) → [16, 4]
/// ones   [1, 8]  ─┘                        │
/// bias_a, bias_b ─── addc(bias_a, bias_b) ─┘
/// ```
///
/// - `mul` is a no-op that `declutter_neutral` (`core/src/ops/binary.rs`)
///   deletes, because one input is uniformly `Mul`'s neutral element.
/// - `addc` has two constant inputs, so it is constant-folded — by `wire_node`
///   or by `PropConst`, depending on `bias_volume`.
/// - `matmul` is an `EinSum`, which `optimize()` lowers to `OptMatMul`.
///
/// The `add` and `matmul` nodes survive every pass and carry the numerics the
/// course asserts on.
pub fn tiny_model(bias_volume: BiasVolume) -> TractResult<TypedModel> {
    let mut model = TypedModel::default();

    let input = model.add_source("input", f32::fact([M, K]))?;
    let ones = model.add_const("ones", Tensor::from_shape(&[1, K], &[1f32; K])?)?;
    let mul_node = model.wire_node("mul", mul(), &[input, ones])?[0];

    let bias_shape = bias_volume.shape();
    let bias_len: usize = bias_shape.iter().product();
    let bias_a =
        model.add_const("bias_a", Tensor::from_shape(bias_shape, &vec![0.25f32; bias_len])?)?;
    let bias_b =
        model.add_const("bias_b", Tensor::from_shape(bias_shape, &vec![0.75f32; bias_len])?)?;
    let addc = model.wire_node("addc", add(), &[bias_a, bias_b])?[0];

    let add_node = model.wire_node("add", add(), &[mul_node, addc])?[0];

    let weights = model.add_const("weights", Tensor::from_shape(&[K, N], &[0.5f32; K * N])?)?;
    let axes: AxesMapping = "ij,jk->ik".parse()?;
    let matmul =
        model.wire_node("matmul", EinSum::new(axes, f32::datum_type()), &[add_node, weights])?[0];

    model.select_output_outlets(&[matmul])?;
    Ok(model)
}

/// Two chained `Add`s against separate constants, the fixture for the
/// hand-written rewrite rule in [`HoistConstantAdds`].
///
/// Neither `Add` is constant-foldable on its own: each has the running value as
/// one input, so `PropConst` cannot touch either. Declutter leaves both in place.
pub fn two_adds_model() -> TractResult<TypedModel> {
    let mut model = TypedModel::default();

    let input = model.add_source("input", f32::fact([M, K]))?;
    let c1 = model.add_const("c1", Tensor::from_shape(&[M, K], &vec![0.25f32; M * K])?)?;
    let c2 = model.add_const("c2", Tensor::from_shape(&[M, K], &vec![0.75f32; M * K])?)?;

    let first = model.wire_node("add_c1", add(), &[input, c1])?[0];
    let second = model.wire_node("add_c2", add(), &[first, c2])?[0];

    model.select_output_outlets(&[second])?;
    Ok(model)
}

/// The variable input and the constant input of a binary node, if exactly one
/// of the two inputs is a constant.
fn split_const(model: &TypedModel, node: &TypedNode) -> Option<(OutletId, OutletId)> {
    let is_const = |o: &OutletId| model.outlet_fact(*o).is_ok_and(|f| f.konst.is_some());
    match (is_const(&node.inputs[0]), is_const(&node.inputs[1])) {
        (false, true) => Some((node.inputs[0], node.inputs[1])),
        (true, false) => Some((node.inputs[1], node.inputs[0])),
        _ => None,
    }
}

/// Re-associate `Add(Add(x, c1), c2)` into `Add(x, Add(c1, c2))`.
///
/// The rule only moves wires; it never does the arithmetic itself. Once both
/// constants feed a single `Add`, that node has no variable input, so the
/// existing `PropConst` pass folds it to one constant on the following round —
/// `Rewriter::rewrite` runs `prop_consts` after each pass, so a single
/// [`HoistConstantAdds`] call is enough to see the `Add` count drop from 2 to 1.
///
/// Requires the inner `Add` to have exactly one successor and to not be a model
/// output. The patch only shunts the outer node; the inner one disappears solely
/// by becoming dead and being dropped by `compact()`. Anything else still holding
/// it — a second consumer, or the output list, which `successors` does not count
/// — keeps it alive, so the rewrite keeps the same op count while trading a
/// shared `x + c1` for a recomputation from `x`; and when `c2` broadcasts against
/// `c1`, the fused constant is larger than the one it replaced.
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

/// Hoists chained constant addends into a single constant.
///
/// Wraps [`hoist_constant_adds`] in a `Rewriter`, which is the shape tract uses
/// for a cross-op pattern (N ops in, M ops out) as opposed to a single-op
/// `Op::declutter`.
#[derive(Debug, Default)]
pub struct HoistConstantAdds;

impl ModelTransform for HoistConstantAdds {
    fn name(&self) -> StaticName {
        "hoist-constant-adds".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        Rewriter::default()
            .with_rule_for::<TypedBinOp>("hoist-constant-adds", hoist_constant_adds)
            .rewrite(&(), model)
    }
}

/// [`tiny_model`] with the row count replaced by the symbol `B`.
///
/// Everything else is identical, so diffing this against
/// `tiny_model(BiasVolume::AboveEagerFoldLimit)` isolates exactly what an
/// unknown dimension costs the optimiser.
pub fn symbolic_model() -> TractResult<TypedModel> {
    let mut model = TypedModel::default();
    let b = model.symbols.sym("B");

    let input = model.add_source("input", f32::fact(dims!(b.clone(), K)))?;
    let ones = model.add_const("ones", Tensor::from_shape(&[1, K], &[1f32; K])?)?;
    let mul_node = model.wire_node("mul", mul(), &[input, ones])?[0];

    let bias_a = model.add_const("bias_a", Tensor::from_shape(&[1, K], &[0.25f32; K])?)?;
    let bias_b = model.add_const("bias_b", Tensor::from_shape(&[1, K], &[0.75f32; K])?)?;
    let addc = model.wire_node("addc", add(), &[bias_a, bias_b])?[0];
    let add_node = model.wire_node("add", add(), &[mul_node, addc])?[0];

    let weights = model.add_const("weights", Tensor::from_shape(&[K, N], &[0.5f32; K * N])?)?;
    let axes: AxesMapping = "ij,jk->ik".parse()?;
    let matmul =
        model.wire_node("matmul", EinSum::new(axes, f32::datum_type()), &[add_node, weights])?[0];

    model.select_output_outlets(&[matmul])?;
    Ok(model)
}

/// Write `model` as an NNEF directory at `dir`, replacing anything already there.
///
/// The CLI recognises a *directory* containing `graph.nnef` as NNEF; pointing it
/// at the bare `graph.nnef` file falls through to the TensorFlow loader instead.
pub fn write_nnef(model: &TypedModel, dir: &std::path::Path) -> TractResult<()> {
    if dir.exists() {
        std::fs::remove_dir_all(dir)?;
    }
    std::fs::create_dir_all(dir.parent().unwrap())?;
    tract_nnef::nnef().write_to_dir(model, dir)
}

/// Count of each op name in the graph, in a stable order.
///
/// This is the course's unit of observation: the same histogram is printed after
/// each pipeline stage, and `tract dump --audit-json` produces a matching one on
/// the CLI side.
pub fn op_histogram(model: &TypedModel) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for node in model.nodes() {
        *counts.entry(node.op().name().to_string()).or_insert(0) += 1;
    }
    counts
}

/// Print a stage banner, the graph, and its op histogram to stdout.
pub fn show_stage(label: &str, model: &TypedModel) {
    println!("\n=== {label} ===");
    println!("{model}");
    println!("op histogram: {:?}", op_histogram(model));
    println!("nodes: {}", model.nodes().len());
}

/// The input the course runs every variant of the model on.
pub fn sample_input() -> TractResult<Tensor> {
    let values: Vec<f32> = (0..(M * K)).map(|i| i as f32 / 10.0).collect();
    Tensor::from_shape(&[M, K], &values)
}

/// Run a model on [`sample_input`] and return the single output tensor.
pub fn run(model: &TypedModel) -> TractResult<Tensor> {
    let plan = SimplePlan::new(model.clone())?;
    let mut outputs = plan.run(tvec!(sample_input()?.into()))?;
    Ok(outputs.remove(0).into_tensor())
}
