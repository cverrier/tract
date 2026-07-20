# Plan: compare f16-sigmoid strategies for non-FP16 arm64

The steps below are self-contained; each can be done in its own session. See the
**Dependency / ordering summary** at the end for how to split them.

## Context

PR #2492 (`perf(linalg): add NEON sigmoid_f16 fallback on non-fp16 arm64`) added
a linalg kernel that runs f16 sigmoid on arm64 cores **without** FEAT_FP16 by
round-tripping through f32: convert f16→f32 into a small L1-resident scratch
(`CHUNK = 256`), run the existing f32 NEON sigmoid, convert back. Conversions
use hand-written NEON asm (`FCVTL`/`FCVTN`), 32-lane unrolled.

The maintainer (kali) asked to see how this compares against three alternatives
before deciding it is the right approach:

1. **In-kernel conversions** ("best-in-class"): fuse convert→sigmoid→convert
   entirely in registers, one pass, no scratch buffer.
2. **tract-core closure**: implement the f16 sigmoid closure as
   `cast_to(f32) → sigmoid → cast_to(f16)`.
3. **tract-core codegen rewrite**: during the codegen phase, replace one f16
   `Sigmoid` node with three nodes: `Cast(f16→f32) → Sigmoid(f32) → Cast(f32→f16)`.

(kali also mentioned chunking to avoid cache spills; the current PR already does
this — `act_f16.rs:11-19,126-132` — so it is **out of scope** here.)

The goal of this plan is an **experiment**, not a product change: build/measure
the candidates, produce a numbers-backed comparison, and recommend one direction.
Only the winner gets turned into production code later, in a separate PR.

## Candidates & how each is measured

| ID | Name | Altitude | How it's benchmarked |
|----|------|----------|----------------------|
| **A** | `neon-f32-roundtrip` | linalg kernel | already exists (`arm64simd_sigmoid_f16_32n`); called by name |
| **B** | `neon-f32-fused` | linalg kernel | **new** fused `.S.j2` kernel; called by name |
| **C** | `core-cast-roundtrip` | tract-core closure | **bench-local proxy**: slice → `Tensor::cast_to::<f32>` → `sigmoid_f32` → `cast_to::<f16>` |
| **D** | `codegen-3op` | tract-core graph | **bench-local proxy**: hand-wired `source(f16)→Cast(f32)→sigmoid()→Cast(f16)` model eval |
| — | `native-fp16` | linalg kernel | reference ceiling on FP16-capable cores (`arm64fp16_sigmoid_f16_8n`) |
| — | `generic` | scalar | reference floor (`generic::sigmoid::HSigmoid8`) |

**Why proxies for C and D:** C and D are two mutually-exclusive edits to the
*same* f16 sigmoid path in core, so they cannot coexist in one binary, and
building production versions of approaches we intend to discard is wasted work.
The proxies reproduce the *compute and memory behaviour* faithfully:
- C's proxy = exactly what the closure body would do (a Tensor cast + the f32
  kernel + a Tensor cast). Uses tract's real `cast_to` so its (likely scalar)
  conversion cost is measured honestly.
- D's proxy = the exact 3-op graph the codegen rewrite would emit, wired by
  hand and run through `into_optimized().into_runnable()`, so the materialized
  full-size f32 intermediate tensor and per-node dispatch overhead are real.

Only **B** needs real production code, because an asm kernel's performance can't
be faked. If B or D wins, its production implementation is a follow-up.

## Methodology

- **Branch:** create `exp/sigmoid-f16-approaches-pr-2492` off the current PR head
  (`perf/add-neon-sigmoid-f16-fallback-on-non-fp16-arm64`), so candidate A and
  its existing bench are already present to extend. This branch is **not for
  merge** — it is the reproducibility artifact referenced from the PR comment.
- **Two bench binaries, one run each:**
  - Kernel-level (`tract-linalg`): A, B, C-proxy, native, generic — all callable
    by name, so a single `cargo bench` run compares them in one binary. Primary,
    most reliable comparison (A vs B vs C).
  - Model-level (`tract-core`): D-proxy (3-op model) vs a 1-op f16 sigmoid model
    (= current dispatch, i.e. A end-to-end). Isolates the graph-level cost of D.
- **Hardware:** verify correctness + that benches run on the local **Apple M3 Pro**
  (out-of-order — do **not** trust its numbers). The authoritative numbers come
  from a **Cortex-A55** (in-order) via `dinghy`, run by the user.

---

## Step 0 — Set up the experiment branch  *(DONE)*

- `git checkout perf/add-neon-sigmoid-f16-fallback-on-non-fp16-arm64`
- `git checkout -b exp/sigmoid-f16-approaches-pr-2492`
- Commit this `PLAN.md`.

---

## Step B — Fused in-register f16 sigmoid kernel  *(DONE)*

Fused NEON kernel: load f16 → `FCVTL`/`FCVTL2` to f32 → verbatim f32 sigmoid
polynomial → `FCVTN`/`FCVTN2` → store f16, one pass, no scratch buffer.

- New `linalg/arm64/arm64simd/arm64simd_sigmoid_f16_4n.S.j2` (16 f16 lanes/iter
  main `.loop4` reusing the f32 `.loop4` body plus a 4-lane `.loop` remainder,
  and its accurate `.float` coeffs; auto-compiled by `build.rs`).
- `arm64simd.rs`: added `use tract_data::half::f16;` and
  `sigmoid_impl!(f16, arm64simd_sigmoid_f16_4n, 4, 8, true);` (the macro
  auto-generates the `sigmoid_frame_tests!` correctness module; reached by name via
  `pub use arm64simd::*`). Dispatch in `arm64.rs` left unchanged — B is bench-only.

Verified on M3: `cargo test -p tract-linalg arm64simd_sigmoid_f16_4n` 6/6 pass;
fmt + clippy clean.

---

## Step C — Extend the kernel-level bench (adds A, B, C, and picks up native/generic)  *(DONE)*

Extended `linalg/benches/sigmoid_f16_arm64.rs` with two ids in the existing size
loop: `neon-f32-fused` (candidate B kernel) and `core-cast-roundtrip` (candidate
C proxy: `cast_to::<f32>` → f32 kernel → `cast_to::<f16>`). All five ids now share
the one `sigmoid_f16` group so a single run prints them side by side. Verified on
M3: compiles, all ids run to completion, fmt clean, no new clippy warnings (M3
numbers ignored per methodology).

---

## Step D — Model-level bench (D-proxy vs current)  *(DONE)*

**Files:**
- New: `core/benches/sigmoid_f16_model.rs`, modeled on
  `core/benches/plan_overhead.rs:15-42`.
- `core/Cargo.toml`: add a `[[bench]] name = "sigmoid_f16_model", harness = false`
  stanza (mirror lines 61-63).

Build two runnable models over an f16 input of each size, and bench `plan.run`:
- **`one-op`** (baseline = A end-to-end): `add_source("x", f16::fact([n]))` →
  `wire_node("s", sigmoid(), &[x])` → `auto_outputs()` → `into_optimized()?.into_runnable()?`.
- **`codegen-3op`** (candidate **D** proxy): `source(f16)` →
  `wire_node("c1", cast(f32::datum_type()), ...)` → `wire_node("s", sigmoid(), ...)`
  → `wire_node("c2", cast(f16::datum_type()), ...)` → `auto_outputs()` →
  `into_optimized()?.into_runnable()?`. This is the exact graph a codegen rewrite
  would emit; `into_optimized()` shows whether the optimizer keeps or collapses it.

Reuse `sigmoid()` from `tract_core::ops::nn`, `cast()` from
`tract_core::ops::cast` (see `wire_cast`, `core/src/ops/cast.rs:8-27`), and
`use tract_core::internal::*;` for `TypedModel`/`tvec!`/`tensor`/`TValue`.

Added `core/benches/sigmoid_f16_model.rs` (`one-op` and `codegen-3op` ids in one
`sigmoid_f16_model` group, over the same three sizes as the kernel bench) and the
`[[bench]]` stanza in `core/Cargo.toml`. Both models are built with
`select_output_outlets` + `into_optimized()?.into_runnable()?`; the 3-op proxy
wires `source(f16) → Cast(f32) → sigmoid() → Cast(f16)`.

**Verify:** `cargo bench -p tract-core --bench sigmoid_f16_model` on the M3 —
compiles, both ids run to completion at all three sizes, fmt clean, no new clippy
warnings. `into_optimized()` keeps the two casts (does not collapse the 3-op graph
back to a single f16 dispatch), so the D proxy measures the real 3-op path.
M3 numbers ignored per methodology.

---

## Step E — Run on the Cortex-A55 and collect results  *(DONE)*

Same `.dinghyignore` trick as the PR description. No `--save-baseline` needed:
all candidates are distinct ids within one criterion group, so a single run per
bench prints them side by side and you read the medians.

```sh
git checkout exp/sigmoid-f16-approaches-pr-2492

# keep the dinghy deploy small
printf '/*\n' > .dinghyignore

# 1) kernel-level comparison: A vs B vs C vs native vs generic
cargo dinghy -d <CORTEX_A55_DEVICE> bench -p tract-linalg \
    --bench sigmoid_f16_arm64

# 2) model-level comparison: D-proxy (3-op) vs current (1-op)
cargo dinghy -d <CORTEX_A55_DEVICE> bench -p tract-core \
    --bench sigmoid_f16_model

# cleanup
rm -f .dinghyignore
```

Run the A55 frequency-locked (`performance` governor) as in the PR. Record the
median `time:`/`thrpt:` per id at each of the three sizes.

### Results

Cortex-A55, in-order, locked at 1908 MHz (`performance` governor), via dinghy.
Medians reported as throughput (Melem/s); higher is better. CIs were tight at
every point (in-order core), so medians are stable.

**Build note (gotcha):** `linalg/build.rs` emits a `rerun-if-changed` per
template it processes, which disables cargo's whole-package change scanning. A
*new* `.S.j2` dropped into the globbed `arm64/arm64simd/` dir therefore does not
trigger a build-script rerun, so a cached cross-build never generates the new
kernel's asm and the link fails with an undefined symbol (candidate B's
`arm64simd_sigmoid_f16_4n`). Forcing a rerun (touch `build.rs`, or
`cargo clean -p tract-linalg`) fixes it. Pre-existing behaviour, not changed here.

**Device note:** the A55 SKU used has FEAT_FP16. So `native-fp16` runs (a real
ceiling), and the model-level `one-op` dispatches to the *native* fp16 kernel
rather than the roundtrip fallback — i.e. `one-op` is **not** a clean "A
end-to-end" row here. The fallback kernels A/B/C are still measured directly by
name (bypassing dispatch, baseline FCVTL/FCVTN only), so they faithfully reflect
the non-fp16 fallback cost on this microarchitecture.

Kernel-level (`sigmoid_f16_arm64`):

| size | generic (floor) | A `neon-f32-roundtrip` | B `neon-f32-fused` | C `core-cast-roundtrip` | native-fp16 (ceiling) |
|------|-----------------|------------------------|--------------------|-------------------------|-----------------------|
| 1024 (L1)     | 15.11 | 139.75 | **159.95** | 53.90 | 665.75 |
| 32768 (L2)    | 15.10 | 139.99 | **159.55** | 57.59 | 657.09 |
| 1048576 (DRAM)| 15.08 | 138.40 | **157.74** | 57.52 | 652.51 |

Model-level (`sigmoid_f16_model`):

| size | one-op (native dispatch on this SKU) | D `codegen-3op` |
|------|--------------------------------------|-----------------|
| 1024 (L1)     | 204.88 | 43.38 |
| 32768 (L2)    | 495.97 | 56.78 |
| 1048576 (DRAM)| 456.69 | 57.06 |

Raw criterion output saved in the run logs; reproduce with the two commands above
on the `exp/sigmoid-f16-approaches-pr-2492` branch.

---

## Step F — Analyze, write up, recommend  *(DONE)*

Build a tradeoff matrix scoring A / B / C / D on:
- **Perf** (A55 medians at L1 / L2 / DRAM sizes; note M3 only as a smoke check),
- **`unsafe` / maintainability** (B adds hand-written asm; C/D add none),
- **Portability** (C/D fix every non-top-ISA arch at once — incl. the x86_64
  non-AVX512 case that today needs its own `linalg/src/x86_64_fma/act_f16.rs`),
- **Generality** (does it extend to tanh/silu/gelu, which `x86_64_fma/act_f16.rs`
  already round-trips).

Deliverables:
- A PR comment on #2492: the matrix + raw criterion output, linking the
  `exp/sigmoid-f16-approaches-pr-2492` branch/commit so kali can reproduce.
  *(Per repo rules: no writes to the GitHub side from this machine — draft the
  comment text for the user to post.)*
- A recommendation: likely either "B's margin over A justifies the asm" or
  "C/D are close enough that the portable core-level approach wins and both
  `act_f16.rs` files can eventually be retired". The winner becomes a separate
  follow-up PR.

### Tradeoff matrix

| axis | A `neon-f32-roundtrip` | B `neon-f32-fused` | C `core-cast` closure | D `codegen-3op` |
|------|------------------------|--------------------|------------------------|-----------------|
| **Perf (A55)** | 138–140 Melem/s | **157–160** (best fallback, +14% vs A) | 54–58 | 43–57 |
| **`unsafe`/maint** | hand-written NEON asm (conversions + f32 kernel + scratch) | **most** asm: a full fused kernel, per op | none (a closure) | none (a graph pass) |
| **Portability** | arm64 only; every non-top-ISA arch needs its own | arm64 only, per-op | **all archs at once** (incl. x86_64 non-AVX512) | **all archs at once** |
| **Generality** | per-op | poor — each of tanh/silu/gelu needs its own fused asm | **any activation** — swap the middle op | **any elementwise op** via a generic rule |

### Findings

- **B is the fastest fallback**, a stable **+14%** over A across L1/L2/DRAM. The
  win comes from never spilling the f32 intermediate to the scratch buffer —
  convert→sigmoid→convert stays in registers in one pass.
- **C and D are ~2.4–2.6× slower than A** and land in the same throughput class
  as each other (54–58 vs 43–57). Both bottleneck on the **same thing**: tract's
  core `Cast` does *scalar* f16↔f32 conversion. Their number is a property of the
  scalar cast, **not** of the closure/graph approach itself.
- **C ≥ D.** D materializes a full-size f32 intermediate (extra memory traffic)
  and pays per-node dispatch, so it trails C at L1 (43 vs 54) and only ties it at
  larger sizes. The closure keeps the roundtrip local/chunked.
- native-fp16 ceiling is ~4.1× above B; irrelevant on the genuinely non-fp16
  cores this path targets, but it bounds how much headroom any fallback leaves.

### Recommendation

The A-vs-B-vs-C/D question decomposes into two independent axes: **how fast is
the f16↔f32 conversion** (NEON vs scalar) and **where the roundtrip lives**
(hand asm vs core). The whole A/B → C/D perf cliff is the *first* axis alone.

- **Do not pursue C or D as-is.** At 2.5× slower than the already-merged A they
  are a regression, purely because they route through the scalar core `Cast`.
- **B (+14% over A) does not clearly justify its cost.** It doubles the
  hand-written asm surface (a full fused kernel *per op*), is arm64-only, and
  does not generalize to tanh/silu/gelu — the other activations
  `x86_64_fma/act_f16.rs` already round-trips. A 14% edge on one op is a weak
  return for that maintenance and generality loss. **Keep A** as the shipping
  arm64 fallback.
- **The strategically right direction is portable C/D _paired with a vectorized
  core f16↔f32 cast_.** A NEON/AVX conversion kernel behind `Tensor::cast_to`
  would lift C (and D) out of the scalar-cast floor toward the A/B band while
  keeping zero per-op asm, fixing **every** non-top-ISA arch and **every**
  activation in one place — and letting both `act_f16.rs` files eventually
  retire. C (the closure) is the better of the two portable shapes: simpler than
  a codegen pass and no full-size intermediate. This is the recommended
  follow-up: **vectorize the core cast, then adopt C**; only fall back to B if a
  specific hot sigmoid path needs the last 14% on arm64 before that lands.

---

## Dependency / ordering summary

- **Step 0** first (branch + commit PLAN.md).
- **Step B**, **Step D** are independent and can each be done in isolation.
- **Step C** (bench file) needs B present for the `neon-f32-fused` row, but the
  `core-cast-roundtrip` row is independent of B.
- **Step E** needs B + C + D landed on the branch.
- **Step F** needs E's numbers.

Suggested split across conversations: {0}, {B + its correctness test}, {C}, {D},
{E + F}.

## Verification (end-to-end)

- Correctness: `cargo test -p tract-linalg` passes, including the new
  `sigmoid_frame_tests!` module for `arm64simd_sigmoid_f16_4n` (Step B).
- Both benches compile and run to completion on the M3
  (`cargo bench -p tract-linalg --bench sigmoid_f16_arm64`,
  `cargo bench -p tract-core --bench sigmoid_f16_model`).
- `cargo fmt --all` and `cargo clippy --workspace` clean before any commit.
- Authoritative A55 numbers collected via the Step E dinghy commands.
