# 03 — The same model through the CLI

## Read

- `doc/cli-recipe.md` §"Model loading" and §"Model import pipeline".
- `doc/pipeline.md` §"How the CLI maps to runtimes".

## Predict

1. You serialise the *decluttered* model to NNEF, then reload it. Does the
   `EinSum` come back as an `EinSum`?
2. Can you serialise the *optimised* model?
3. `tract --pass type learn/models/tiny dump` — the model is NNEF. What happens?

## Run

```sh
cargo run -p tract-learn --bin lesson03
```

That writes `learn/models/tiny/`:

```
graph.nnef  addc.dat  weights.dat
```

The whole graph, weights excluded, is five lines:

```
version 1.0;
extension tract_registry tract_core;

graph network(input) -> (matmul) {
  input = external(shape = [16, 8]);
  addc = variable<scalar>(label = "addc", shape = [16, 8]);
  add = add(input, addc);
  weights = variable<scalar>(label = "weights", shape = [8, 4]);
  matmul = matmul(add, weights, transposeA = false, transposeB = false);
}
```

Prediction 1: **the `EinSum` came back as `matmul`.** Serialisation runs
`rewrite_einsum_to_prefix_matmul` (`nnef/src/ser.rs`) first, so a plain 2-D
einsum lands as stock NNEF `matmul` rather than the `tract_core_einsum`
extension fragment. This is a round-trip you can hand-edit: `graph.nnef` is text.

Prediction 2, from the same run:

```
serializing the OPTIMIZED model fails, as it must:
  Translating model to AST: translating node #2 "add" OptAddUnicast:
  No serializer found for node #2 "add" OptAddUnicast
```

There is no NNEF serialiser for any `Opt*` op, by design — grep `nnef/src/` for
`OptMatMul` and you get zero hits. "Optimised is not portable" is not a
convention, it is enforced by the absence of code.

### Now drive the stages from the CLI

```sh
T=./target/debug/tract

$T learn/models/tiny dump                 # default: decluttered
$T -O learn/models/tiny dump              # optimised
```

Default (`Const` nodes are hidden unless you pass `--const`):

```
⓪ 0 Source input
┃   ━━━ 16,8,F32
┣┻ 2 Add add_1
┣┻ 4 EinSum matmul
⓿   ━━━ 16,4,F32
```

With `-O`:

```
⓪ 0 Source input
┣┻ 2 OptAddUnicast add_1
┣ 3 OptMatMulPack matmul.pack_a
┃   ━━━ ,F32 🔍 DynPackedExoticFact { k: Val(8), mn: Val(16), packers: [PackedF32[8]@16+1] }
┣ 4 OptMatMul matmul
⓿   ━━━ 16,4,F32
```

Same two graphs as Lesson 02. Confirm it mechanically rather than by eye —
`--audit-json` is the machine-readable dump, and the text output is explicitly
not meant to be parsed:

```sh
$T learn/models/tiny dump --audit-json |
  python3 -c "import json,sys,collections;print(dict(sorted(collections.Counter(n['op'] for n in json.load(sys.stdin)['nodes']).items())))"
```

```
declutter: {'Add': 1, 'Const': 2, 'EinSum': 1, 'Source': 1}
optimized: {'Const': 1, 'OptAddUnicast': 1, 'OptMatMul': 1, 'OptMatMulPack': 1, 'Source': 1}
```

Identical to the Rust-side histograms. Two views of one pipeline.

### Prediction 3: stages that don't exist for your format

```sh
$T --pass type learn/models/tiny dump
```

```
ERROR Stage type is skipped, it can not be used as stop with these input format or parameters.
```

`analyse`, `incorporate` and `type` are `InferenceModel` stages. NNEF carries
fully resolved shapes and types, so it loads *straight to* `TypedModel` and those
three stages never run. This is the concrete meaning of "loading from NNEF is one
step shorter" in `doc/pipeline.md` — and why NNEF is the recommended deployment
format. To see those stages you need an ONNX model.

## Diffing two stages numerically

The real reason to care about stage boundaries: did optimising change the
answers?

```sh
$T -O learn/models/tiny compare --stage declutter --allow-random-input
```

```
4 node(s) passed the comparison.
```

`compare --stage` runs the decluttered model, keeps every intermediate tensor,
then walks the optimised model node by node checking each output against it.
By default it is *non-cumulative*: each node's output is reset to the reference
before feeding downstream, so you see the **first** divergence rather than an
avalanche. `--cumulative` turns that off when you want to see error accumulate.

This is the tool for "my model got 3% less accurate after `-O`". In-tree
precedent: `harness/nnef-test-cases/conv-with-batch/runme.sh` is exactly this
command.

## Two traps in the flags

**`--pass optimize` does not optimize.**

```sh
$T learn/models/tiny --pass optimize dump --audit-json   # {'Add': 1, 'EinSum': 1, ...}
$T -O learn/models/tiny dump --audit-json                # {'OptMatMul': 1, ...}
```

`--pass optimize` is accepted, runs, and silently gives you the *unoptimised*
graph. Optimisation is chosen by runtime selection, which keys off `-O` /
`--metal` / `--cuda` / `--runtime` and never looks at `--pass`. Lesson 08 comes
back to this.

**`--nnef-tract-core` is a no-op.** `doc/cli-recipe.md` tells you to pass it when
reloading NNEF. That advice is stale: the flag is explicitly
`"no-op, kept for backward compatibility"` and hidden from `--help`
(`cli/src/main.rs`); the `tract_core` registry is on by default. The live flag is
the negative one, `--no-nnef-tract-core`.

## Exercise

Hand-edit `learn/models/tiny/graph.nnef` — change `add` to `sub`, or the shapes —
and reload it. Then try `dump --nnef-graph -`, which prints the graph text
tract *itself* would emit. Diff that against the file you wrote. This is the
`expected`/`found` pattern used across `harness/nnef-test-cases/`.

---

Next: [04 — Reading the pass lists](04-pass-lists.md)
