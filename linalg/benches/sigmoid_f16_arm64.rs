// Microbenchmark: arm64 f16 sigmoid across the cache hierarchy.
// The three sizes span working sets from cache-resident (compute-bound) to
// larger-than-cache (bandwidth-bound), so the generic-vs-roundtrip comparison
// holds in both regimes. Exact cache tiers are hardware-specific — these are
// representative, not calibrated to a particular core.
// Kernels are called by name to bypass dispatch, which always selects the
// native fp16 kernel on an fp16-capable core. The f32-roundtrip fallback only
// uses baseline FCVTL/FCVTN, so it is safe to call on any arm64.
#![cfg(target_arch = "aarch64")]

use criterion::*;
use tract_data::prelude::*;
use tract_linalg::element_wise::ElementWiseKer;

fn aligned_input(n: usize) -> Tensor {
    let mut t = unsafe { Tensor::uninitialized_aligned::<f16>(&[n], 16).unwrap() };
    let s = unsafe { t.as_slice_mut_unchecked::<f16>() };
    for (i, x) in s.iter_mut().enumerate() {
        *x = f16::from_f32((i as f32 / 10.0).sin() * 5.0);
    }
    t
}

fn sigmoid_f16(c: &mut Criterion) {
    for n in [1024usize, 32_768, 1_048_576] {
        let mut group = c.benchmark_group("sigmoid_f16");
        group.throughput(Throughput::Elements(n as u64));

        let mut tg = aligned_input(n);
        let sg = unsafe { tg.as_slice_mut_unchecked::<f16>() };
        group.bench_with_input(BenchmarkId::new("generic", n), &(), |b, _| {
            b.iter(|| tract_linalg::generic::sigmoid::HSigmoid8::run(sg, ()))
        });

        let mut tf = aligned_input(n);
        let sf = unsafe { tf.as_slice_mut_unchecked::<f16>() };
        group.bench_with_input(BenchmarkId::new("neon-f32-roundtrip", n), &(), |b, _| {
            b.iter(|| tract_linalg::arm64::arm64simd_sigmoid_f16_4n::run(sf, ()))
        });

        let mut tb = aligned_input(n);
        let sb = unsafe { tb.as_slice_mut_unchecked::<f16>() };
        group.bench_with_input(BenchmarkId::new("neon-f32-fused", n), &(), |b, _| {
            b.iter(|| tract_linalg::arm64::arm64simd_sigmoid_f16_4n_fused::run(sb, ()))
        });

        // Candidate C proxy: the tract-core f16 sigmoid closure body, i.e.
        // Tensor::cast_to::<f32> → f32 kernel → Tensor::cast_to::<f16>. The casts
        // are tract's real (scalar) conversions, so their cost is measured honestly.
        let tc = aligned_input(n);
        group.bench_with_input(BenchmarkId::new("core-cast-roundtrip", n), &(), |b, _| {
            b.iter(|| {
                let mut f32t = tc.cast_to::<f32>().unwrap().into_owned();
                let s = unsafe { f32t.as_slice_mut_unchecked::<f32>() };
                tract_linalg::arm64::arm64simd_sigmoid_f32_4n::run(s, ());
                f32t.cast_to::<f16>().unwrap().into_owned()
            })
        });

        if tract_linalg::arm64::has_fp16() {
            let mut tn = aligned_input(n);
            let sn = unsafe { tn.as_slice_mut_unchecked::<f16>() };
            group.bench_with_input(BenchmarkId::new("native-fp16", n), &(), |b, _| {
                b.iter(|| tract_linalg::arm64::arm64fp16_sigmoid_f16_8n::run(sn, ()))
            });
        }
        group.finish();
    }
}

criterion_group!(g, sigmoid_f16);
criterion_main!(g);
