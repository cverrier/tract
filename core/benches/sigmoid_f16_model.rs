//! Model-level f16-sigmoid bench: candidate D (codegen 3-op rewrite) vs current.
//!
//! Builds two runnable models over an f16 input and measures `plan.run()`:
//! - `one-op`: a single f16 `Sigmoid` node — the current dispatch, i.e. candidate
//!   A end-to-end (the linalg fallback kernel, reached via the sigmoid closure).
//! - `codegen-3op`: `Cast(f16→f32) → Sigmoid(f32) → Cast(f32→f16)` — the exact
//!   graph a codegen rewrite (candidate D) would emit. Running it through
//!   `into_optimized()` shows whether the optimizer keeps or collapses the casts,
//!   and materializes the full-size f32 intermediate so its cost is real.

use criterion::*;
use tract_core::internal::*;
use tract_core::ops::cast::cast;
use tract_core::ops::nn::sigmoid;
use tract_data::half::f16;

fn f16_input(n: usize) -> TValue {
    let v: Vec<f16> = (0..n).map(|i| f16::from_f32((i as f32 / 10.0).sin() * 5.0)).collect();
    tensor1(&v).into()
}

fn one_op(n: usize) -> Arc<TypedRunnableModel> {
    let mut model = TypedModel::default();
    let x = model.add_source("x", f16::fact([n])).unwrap();
    let s = model.wire_node("s", sigmoid(), &[x]).unwrap();
    model.select_output_outlets(&s).unwrap();
    model.into_optimized().unwrap().into_runnable().unwrap()
}

fn codegen_3op(n: usize) -> Arc<TypedRunnableModel> {
    let mut model = TypedModel::default();
    let x = model.add_source("x", f16::fact([n])).unwrap();
    let c1 = model.wire_node("c1", cast(f32::datum_type()), &[x]).unwrap();
    let s = model.wire_node("s", sigmoid(), &c1).unwrap();
    let c2 = model.wire_node("c2", cast(f16::datum_type()), &s).unwrap();
    model.select_output_outlets(&c2).unwrap();
    model.into_optimized().unwrap().into_runnable().unwrap()
}

fn sigmoid_f16_model(c: &mut Criterion) {
    for n in [1024usize, 32_768, 1_048_576] {
        let mut group = c.benchmark_group("sigmoid_f16_model");
        group.throughput(Throughput::Elements(n as u64));
        let input = f16_input(n);

        let one = one_op(n);
        group.bench_with_input(BenchmarkId::new("one-op", n), &(), |b, _| {
            b.iter(|| one.run(tvec![input.clone()]).unwrap())
        });

        let three = codegen_3op(n);
        group.bench_with_input(BenchmarkId::new("codegen-3op", n), &(), |b, _| {
            b.iter(|| three.run(tvec![input.clone()]).unwrap())
        });

        group.finish();
    }
}

criterion_group!(g, sigmoid_f16_model);
criterion_main!(g);
