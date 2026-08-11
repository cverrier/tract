//! Executable form of every claim the lessons make.
//!
//! Each lesson's markdown quotes op histograms and node counts. This file
//! asserts them, so a tract change that invalidates a lesson breaks the build
//! instead of quietly making the prose wrong.

use std::collections::BTreeMap;

use tract_core::internal::*;
use tract_core::optim::Optimizer;
use tract_core::transform::ModelTransform;
use tract_learn::*;

fn histogram(pairs: &[(&str, usize)]) -> BTreeMap<String, usize> {
    pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
}

/// Asserts [`HoistConstantAdds`] leaves `model` untouched, comparing node names
/// rather than the op histogram: on a fixture where the inner `Add` survives, the
/// rewrite replaces one `Const` and one `Add` with one of each, so the counts
/// match even when the rule wrongly fires.
fn assert_rule_declines(model: &TypedModel, reason: &str) -> TractResult<()> {
    let names =
        |m: &TypedModel| -> Vec<String> { m.nodes().iter().map(|n| n.name.clone()).collect() };
    let mut after = model.clone();
    HoistConstantAdds.transform(&mut after)?;
    assert_eq!(names(model), names(&after), "{reason}");
    Ok(())
}

#[test]
fn lesson01_output_is_hand_computable() -> TractResult<()> {
    let model = tiny_model(BiasVolume::AboveEagerFoldLimit)?;
    let output = run(&model)?;

    // input[r, c] = (r * K + c) / 10 ; mul by 1 is a no-op ; bias adds 0.25 + 0.75 = 1
    // output[r, n] = sum over c of (input[r, c] + 1) * 0.5
    let expected: Vec<f32> = (0..M)
        .flat_map(|r| {
            let row: f32 = (0..K).map(|c| ((r * K + c) as f32 / 10.0) + 1.0).sum::<f32>() * 0.5;
            std::iter::repeat_n(row, N)
        })
        .collect();

    output.close_enough(&Tensor::from_shape(&[M, N], &expected)?, false)?;
    Ok(())
}

#[test]
fn lesson02_stage_histograms() -> TractResult<()> {
    let typed = tiny_model(BiasVolume::AboveEagerFoldLimit)?;
    assert_eq!(
        op_histogram(&typed),
        histogram(&[("Add", 2), ("Const", 4), ("EinSum", 1), ("Mul", 1), ("Source", 1)])
    );
    assert_eq!(typed.nodes().len(), 9);

    let decluttered = typed.clone().into_decluttered()?;
    assert_eq!(
        op_histogram(&decluttered),
        histogram(&[("Add", 1), ("Const", 2), ("EinSum", 1), ("Source", 1)])
    );
    assert_eq!(decluttered.nodes().len(), 5);

    let optimized = decluttered.clone().into_optimized()?;
    assert_eq!(
        op_histogram(&optimized),
        histogram(&[
            ("Const", 1),
            ("OptAddUnicast", 1),
            ("OptMatMul", 1),
            ("OptMatMulPack", 1),
            ("Source", 1),
        ])
    );

    let reference = run(&typed)?;
    run(&decluttered)?.close_enough(&reference, false)?;
    run(&optimized)?.close_enough(&reference, false)?;
    Ok(())
}

#[test]
fn lesson02_small_constants_are_folded_before_declutter_runs() -> TractResult<()> {
    let eager = tiny_model(BiasVolume::BelowEagerFoldLimit)?;

    // `addc` never became a node: wire_node evaluated it, and the resulting
    // all-ones tensor was deduplicated onto the existing `ones` const.
    assert!(!eager.nodes().iter().any(|n| n.name == "addc"));
    assert_eq!(eager.nodes().iter().filter(|n| n.op().name() == "Add").count(), 1);

    // bias_a and bias_b survive wiring as dead nodes, with no successors.
    for dead in ["bias_a", "bias_b"] {
        let node = eager.nodes().iter().find(|n| n.name == dead).unwrap();
        assert!(node.outputs[0].successors.is_empty(), "{dead} should be dangling");
    }

    // The volume-128 variant keeps `addc` alive until PropConst folds it.
    let lazy = tiny_model(BiasVolume::AboveEagerFoldLimit)?;
    assert!(lazy.nodes().iter().any(|n| n.name == "addc" && n.op().name() == "Add"));

    // Both variants converge on the same decluttered shape.
    assert_eq!(op_histogram(&eager.into_decluttered()?), op_histogram(&lazy.into_decluttered()?));
    Ok(())
}

#[test]
fn lesson04_declutter_fires_one_patch_at_a_time() -> TractResult<()> {
    let at = |steps: usize| -> TractResult<BTreeMap<String, usize>> {
        let mut model = tiny_model(BiasVolume::AboveEagerFoldLimit)?;
        Optimizer::declutter().stopping_at(steps).session().optimize(&mut model)?;
        Ok(op_histogram(&model))
    };

    assert_eq!(at(1)?["Mul"], 1, "patch 1 folds addc, the Mul is still there");
    assert_eq!(at(1)?["Add"], 1);
    assert!(!at(2)?.contains_key("Mul"), "patch 2 removes the neutral Mul");
    assert_eq!(at(2)?, at(4)?, "declutter has reached its fixpoint by patch 2");
    Ok(())
}

#[test]
fn lesson04_codegen_reveals_intermediate_ops() -> TractResult<()> {
    let decluttered = tiny_model(BiasVolume::AboveEagerFoldLimit)?.into_decluttered()?;
    let at = |steps: usize| -> TractResult<BTreeMap<String, usize>> {
        let mut model = decluttered.clone();
        Optimizer::codegen().stopping_at(steps).session().optimize(&mut model)?;
        Ok(op_histogram(&model))
    };

    // EinSumMatMul exists only between two codegen patches; it is never visible
    // in a fully optimized graph.
    assert_eq!(at(2)?["EinSumMatMul"], 1);
    assert!(!at(4)?.contains_key("EinSumMatMul"));

    // Two packing nodes appear, then the constant-weight one is baked into
    // OptMatMul itself and compacted away.
    assert_eq!(at(3)?["OptMatMulPack"], 2);
    assert_eq!(at(4)?["OptMatMulPack"], 1);
    Ok(())
}

#[test]
fn lesson05_hoisting_rule_removes_one_add() -> TractResult<()> {
    let decluttered = two_adds_model()?.into_decluttered()?;
    assert_eq!(op_histogram(&decluttered)["Add"], 2, "stock declutter does not fold this");

    let mut hoisted = decluttered.clone();
    HoistConstantAdds.transform(&mut hoisted)?;

    assert_eq!(
        op_histogram(&hoisted),
        histogram(&[("Add", 1), ("Const", 1), ("Source", 1)]),
        "the rule reassociates, then prop_consts folds the two constants into one"
    );
    run(&hoisted)?.close_enough(&run(&decluttered)?, false)?;
    Ok(())
}

#[test]
fn lesson05_rule_leaves_shared_inner_add_alone() -> TractResult<()> {
    // Same chain, but a Mul also consumes the inner Add, so hoisting it would
    // duplicate work rather than remove a node.
    let mut model = TypedModel::default();
    let input = model.add_source("input", f32::fact([M, K]))?;
    let c1 = model.add_const("c1", Tensor::from_shape(&[M, K], &vec![0.25f32; M * K])?)?;
    let c2 = model.add_const("c2", Tensor::from_shape(&[M, K], &vec![0.75f32; M * K])?)?;
    let first = model.wire_node("add_c1", tract_core::ops::math::add(), &[input, c1])?[0];
    let second = model.wire_node("add_c2", tract_core::ops::math::add(), &[first, c2])?[0];
    let other = model.wire_node("mul_c2", tract_core::ops::math::mul(), &[first, c2])?[0];
    let out = model.wire_node("out", tract_core::ops::math::add(), &[second, other])?[0];
    model.select_output_outlets(&[out])?;

    assert_rule_declines(&model, "single-successor guard should block the rule")
}

#[test]
fn lesson05_rule_leaves_inner_add_that_is_an_output_alone() -> TractResult<()> {
    // The inner Add has one successor but is also a model output, so it survives
    // compaction and the hoist buys nothing. `successors` does not count the
    // output list, so this needs a guard of its own.
    let mut model = TypedModel::default();
    let input = model.add_source("input", f32::fact([M, K]))?;
    let c1 = model.add_const("c1", Tensor::from_shape(&[M, K], &vec![0.25f32; M * K])?)?;
    let c2 = model.add_const("c2", Tensor::from_shape(&[M, K], &vec![0.75f32; M * K])?)?;
    let first = model.wire_node("add_c1", tract_core::ops::math::add(), &[input, c1])?[0];
    let second = model.wire_node("add_c2", tract_core::ops::math::add(), &[first, c2])?[0];
    model.select_output_outlets(&[first, second])?;

    assert_rule_declines(&model, "model-output guard should block the rule")
}

#[test]
fn lesson06_symbolic_optimizes_and_binds() -> TractResult<()> {
    let symbolic = symbolic_model()?;
    let b = symbolic.symbols.sym("B");

    // An unknown row count does not stop EinSum from lowering: only K and N need
    // to be concrete for a kernel to be chosen.
    let optimized = symbolic.clone().into_optimized()?;
    assert_eq!(optimized.nodes().iter().filter(|n| n.op().name() == "OptMatMul").count(), 1);
    assert_eq!(optimized.output_fact(0)?.shape[0], b.to_dim());

    // Binding B then optimizing must agree with the born-concrete model.
    let bound = symbolic
        .into_decluttered()?
        .set_symbols(&[(b, M.to_dim())].into_iter().collect())?
        .into_optimized()?;
    assert_eq!(bound.output_fact(0)?.shape.as_concrete().unwrap(), &[M, N]);

    let concrete = tiny_model(BiasVolume::BelowEagerFoldLimit)?.into_optimized()?;
    run(&bound)?.close_enough(&run(&concrete)?, false)?;
    Ok(())
}

#[test]
fn lesson06_symbol_is_not_what_blocks_the_unicast_add() -> TractResult<()> {
    // The plain Add in the symbolic graph is explained by the bias shape, not by
    // B being unknown: the same-shaped concrete model behaves identically, and
    // only the [M, K] bias reaches the unicast codegen's 32-element threshold.
    let symbolic = op_histogram(&symbolic_model()?.into_optimized()?);
    let narrow = op_histogram(&tiny_model(BiasVolume::BelowEagerFoldLimit)?.into_optimized()?);
    let wide = op_histogram(&tiny_model(BiasVolume::AboveEagerFoldLimit)?.into_optimized()?);

    assert_eq!(symbolic, narrow);
    assert_eq!(symbolic["Add"], 1);
    assert_eq!(wide["OptAddUnicast"], 1);
    assert!(!wide.contains_key("Add"));
    Ok(())
}

#[test]
fn lesson03_decluttered_serializes_but_optimized_does_not() -> TractResult<()> {
    let decluttered = tiny_model(BiasVolume::AboveEagerFoldLimit)?.into_decluttered()?;
    let dir = std::env::temp_dir().join("tract-learn-test-nnef");
    write_nnef(&decluttered, &dir)?;

    let graph = std::fs::read_to_string(dir.join("graph.nnef"))?;
    // The EinSum is rewritten to a prefix matmul during serialization, so it
    // lands as stock NNEF `matmul`, not the tract_core_einsum extension.
    assert!(graph.contains("matmul(add, weights"), "got:\n{graph}");
    assert!(!graph.contains("tract_core_einsum"));

    let optimized = decluttered.into_optimized()?;
    let err = write_nnef(&optimized, &std::env::temp_dir().join("tract-learn-test-nnef-opt"))
        .unwrap_err();
    assert!(
        format!("{err:#}").contains("No serializer found"),
        "optimized graphs are machine-specific and must not serialize: {err:#}"
    );

    std::fs::remove_dir_all(&dir)?;
    Ok(())
}
