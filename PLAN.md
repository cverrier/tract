# f16-sigmoid strategies for non-FP16 arm64 (PR #2492) — results

**Status: complete.** Experiment on branch `exp/sigmoid-f16-approaches-pr-2492`
(not for merge; reproducibility artifact). Only the winner becomes a follow-up PR.

## Context

PR #2492 added a linalg kernel running f16 sigmoid on arm64 cores **without**
FEAT_FP16 by round-tripping through f32: convert f16→f32 into an L1-resident
scratch (`CHUNK = 256`), run the f32 NEON sigmoid, convert back. kali asked how
this compares to three alternatives before accepting it.

## Candidates

| ID | Name | Altitude | Measured via |
|----|------|----------|--------------|
| **A** | `neon-f32-roundtrip` | linalg kernel | already exists (`arm64simd_sigmoid_f16_32n`) |
| **B** | `neon-f32-fused` | linalg kernel | new fused kernel: convert→sigmoid→convert in registers, one pass, no scratch |
| **C** | `core-cast-roundtrip` | tract-core closure | proxy: `cast_to::<f32>` → f32 kernel → `cast_to::<f16>` |
| **D** | `codegen-3op` | tract-core graph | proxy: `source(f16)→Cast(f32)→sigmoid→Cast(f16)`, optimized+run |
| — | `native-fp16` | linalg kernel | ceiling (`arm64fp16_sigmoid_f16_8n`) |
| — | `generic` | scalar | floor (`generic::sigmoid::HSigmoid8`) |

C/D use proxies because they are mutually-exclusive edits to the *same* core path
(can't coexist) and building versions we'll discard is wasted work; the proxies
reproduce the real compute/memory behaviour (real `cast_to`, real 3-op graph).
Only **B** needed real production code (asm perf can't be faked).

## Results

Cortex-A55 (in-order), locked 1908 MHz (`performance` governor), via dinghy.
Throughput in Melem/s, higher better; CIs tight. **M3 numbers ignored**
(out-of-order — smoke check only).

**Device note:** this A55 SKU *has* FEAT_FP16, so model-level `one-op` dispatches
to the native fp16 kernel, i.e. it is **not** a clean "A end-to-end" row. The
fallback kernels A/B/C are measured directly by name (bypassing dispatch), so
they faithfully reflect the non-fp16 fallback cost.

**Build gotcha:** `linalg/build.rs` emits `rerun-if-changed` per template, which
disables cargo's package change scanning — a *new* `.S.j2` doesn't trigger a
build-script rerun, so a cached cross-build never generates the kernel and the
link fails (undefined `arm64simd_sigmoid_f16_4n`). Fix: touch `build.rs` or
`cargo clean -p tract-linalg`. Pre-existing behaviour.

Kernel-level (`sigmoid_f16_arm64`):

| size | generic | A roundtrip | B fused | C core-cast | native-fp16 |
|------|---------|-------------|---------|-------------|-------------|
| 1024 (L1)      | 15.11 | 139.75 | **159.95** | 53.90 | 665.75 |
| 32768 (L2)     | 15.10 | 139.99 | **159.55** | 57.59 | 657.09 |
| 1048576 (DRAM) | 15.08 | 138.40 | **157.74** | 57.52 | 652.51 |

Model-level (`sigmoid_f16_model`):

| size | one-op (native on this SKU) | D codegen-3op |
|------|-----------------------------|---------------|
| 1024 (L1)      | 204.88 | 43.38 |
| 32768 (L2)     | 495.97 | 56.78 |
| 1048576 (DRAM) | 456.69 | 57.06 |

Reproduce (A55 frequency-locked, `performance` governor):

```sh
git checkout exp/sigmoid-f16-approaches-pr-2492

printf '/*\n' > .dinghyignore   # keep the dinghy deploy small

# 1) kernel-level: A vs B vs C vs native vs generic (one group, read the medians)
cargo dinghy -d <CORTEX_A55_DEVICE> bench -p tract-linalg --bench sigmoid_f16_arm64

# 2) model-level: D-proxy (3-op) vs one-op
cargo dinghy -d <CORTEX_A55_DEVICE> bench -p tract-core --bench sigmoid_f16_model

rm -f .dinghyignore
```

M3 smoke check (numbers ignored): same two `cargo bench` commands without dinghy.

## Tradeoff matrix

| axis | A roundtrip | B fused | C closure | D codegen |
|------|-------------|---------|-----------|-----------|
| **Perf (A55)** | 138–140 | **157–160** (+14% vs A) | 54–58 | 43–57 |
| **unsafe/maint** | hand asm (conv+kernel+scratch) | **most** asm — full fused kernel per op | none | none |
| **Portability** | arm64 only, per-arch | arm64 only, per-op | **all archs** (incl. x86_64 non-AVX512) | **all archs** |
| **Generality** | per-op | poor (each of tanh/silu/gelu needs its own asm) | **any activation** | **any elementwise op** |

## Findings & recommendation

- **B is the fastest fallback**, stable **+14%** over A (never spills the f32
  intermediate to scratch).
- **C/D are ~2.5× slower than A**, both bottlenecked on the *same* thing: tract's
  core `Cast` does **scalar** f16↔f32 conversion. That number is a property of
  the scalar cast, not the closure/graph shape. D ≤ C (materializes a full-size
  f32 intermediate + per-node dispatch).

The A/B-vs-C/D cliff is one axis alone: **conversion speed** (NEON vs scalar),
separate from **where the roundtrip lives** (asm vs core).

- **Don't pursue C/D as-is** — 2.5× regression, purely from the scalar core cast.
- **B's +14% doesn't justify its cost** — doubles hand-asm surface (a fused
  kernel *per op*), arm64-only, doesn't generalize to tanh/silu/gelu. **Keep A**
  as the shipping arm64 fallback.
- **Strategic direction: vectorize the core f16↔f32 cast, then adopt C.** A
  NEON/AVX conversion behind `Tensor::cast_to` lifts C (and D) toward the A/B
  band with zero per-op asm, fixing every non-top-ISA arch and every activation
  at once (both `act_f16.rs` files can retire). C > D (simpler, no full-size
  intermediate). Fall back to B only if a specific hot path needs the last 14%.

**Deliverable:** PR #2492 comment with the matrix + raw criterion output, linking
this branch/commit. Per repo rules, draft the text for the user to post — no
GitHub writes from this machine.
