# 07 — Metal dispatch

## Read

- `doc/pipeline.md` §"The Runtime trait" and the runtime table. Two paragraphs.

## The idea

Everything up to `declutter` is target-independent. Everything after belongs to a
`Runtime`:

```rust
pub trait Runtime {
    fn name(&self) -> StaticName;
    fn prepare(&self, model: TypedModel) -> TractResult<Box<dyn Runnable>>;
}
```

`DefaultRuntime` (registered as `cpu`, aliased `default`) does
`model.into_optimized()` then builds a plan. `MetalRuntime` does one extra thing
first:

```rust
// metal/src/lib.rs
MetalTransform::default().transform(&mut model)?;
model = model.into_optimized()?;
```

So Metal is a `ModelTransform` that swaps in device ops, followed by the *same*
`into_optimized()` you have been watching, which lowers whatever the transform
left on the CPU. The runtime registry is `inventory`-based, so any backend crate
linked into the binary contributes — that is what `list-runtimes` prints.

## Predict

The decluttered graph from Lesson 03 is `Source → Add → EinSum`. On CPU with `-O`
it became `OptAddUnicast`, `OptMatMulPack`, `OptMatMul`.

1. What replaces `Add` and `EinSum` under `--metal`?
2. The input arrives in host memory and the caller expects host memory back.
   What has to appear in the graph that was not needed on CPU?
3. Does `--metal` need `-O`?

## Run

```sh
./target/debug/tract --metal learn/models/tiny dump
```

```
⓪ 0 Source input
┃   ━━━ 16,8,F32
┣ 1 DeviceSyncToDevice add_1.to-device-0
┃   ━━━ 16,8,F32 🔍 FromHost(16,8,F32)
┣┻ 3 MetalAdd add_1
┃   ━━━ 16,8,F32 🔍 FromDevice(16,8,F32)
┣┻ 5 MetalMlxGemm matmul
┃   ━━━ 16,4,F32 🔍 FromDevice(16,4,F32)
┣ 6 DeviceSyncToHost matmul.to-host-0-out
⓿   ━━━ 16,4,F32
```

Histograms side by side:

```
CPU   -O : {'Const': 1, 'OptAddUnicast': 1, 'OptMatMul': 1, 'OptMatMulPack': 1, 'Source': 1}
Metal    : {'Const': 2, 'DeviceSyncToDevice': 1, 'DeviceSyncToHost': 1, 'MetalAdd': 1, 'MetalMlxGemm': 1, 'Source': 1}
```

Two ops, two completely different lowerings, from one portable graph.

- `Add` → **`MetalAdd`**, `EinSum` → **`MetalMlxGemm`** (the MLX GEMM
  implementation; `metal/src/transform.rs` can also select MFA or GGML kernels).
- Prediction 2: **`DeviceSyncToDevice` and `DeviceSyncToHost`** appear, bracketing
  the GPU region. The facts between them read `FromHost(...)` / `FromDevice(...)`,
  which is the residency of the tensor made visible in the graph. Those two nodes
  are the host/device copies, and they are exactly what you want to count when a
  GPU model is unexpectedly slow: a graph that ping-pongs shows many of them.
- Note there is no packing node. Packing is a CPU micro-kernel concern; the Metal
  kernels take their own layouts.

Prediction 3: **no `-O` needed.** Runtime selection tests `--metal` *before* it
tests `-O` (`cli/src/params.rs`), and `MetalRuntime::prepare` optimises
internally. `-O --metal` and `--metal` give the same graph. This is the one place
where leaving off `-O` is not a mistake — contrast Lesson 03's trap, where
`--pass optimize` without `-O` silently gave an unoptimised graph.

## Which ops actually moved

For a real model the interesting question is what *stayed* on the CPU. Diff the
op sets:

```sh
T=./target/debug/tract
for rt in "" "--metal"; do
  $T $rt learn/models/tiny dump --audit-json |
    python3 -c "import json,sys;print(sorted({n['op'] for n in json.load(sys.stdin)['nodes']}))"
done
```

On a large model, any op with no Metal implementation stays as its CPU form and
forces a sync pair around itself. That is the first thing to look at when a
backend port underperforms: not kernel speed, but how many times the data crossed
the bus. `dump --profile` marks CPU-fallback nodes in yellow.

## Not on a Mac?

`list-runtimes` will show `cpu`, `unoptimized`, and `cuda` if it was compiled in.
The structure is identical — `CudaTransform` then `optimize()` — so read this
lesson with `--cuda`, and expect `Cuda*` ops with the same sync-node bracketing.
With neither, `--metal` fails at runtime lookup with
`Runtime 'metal' not found`, which is itself a useful thing to have seen.

## Exercise

Add a pooling op to a model and compare its CPU and Metal lowerings. Then find an
op with *no* Metal implementation, put it in the middle of two Metal-friendly ops,
and count the `DeviceSync*` nodes. Predict the count before you dump.

---

Next: [08 — Your first contribution](08-first-contribution.md)
