# A lab manual for the tract compilation pipeline

Nine short lessons that take one deliberately tiny model through every stage
between `TypedModel::default()` and a running plan, and make you look at what
changed at each step.

This is **not** a second copy of the conceptual docs. `doc/pipeline.md`,
`doc/symbolic-shapes.md`, `doc/graph.md` and `doc/op.md` already explain the
ideas well, and every lesson starts by pointing you at the relevant section.
What was missing was practice. So each lesson is:

1. **Read** — a named section of `doc/`. Short.
2. **Predict** — write your answer down *before* running anything. This is the
   part that does the work; skipping it turns the course into reading.
3. **Run** — a Rust binary (`cargo run -p tract-learn --bin lessonNN`), then the
   equivalent `tract` CLI invocation. Lessons 07 and 08 are CLI-only.
4. **Explain** — map what you saw onto the source, with the file to open.

## The lessons

| # | File | What you come away able to do |
|---|---|---|
| 00 | [00-setup.md](lessons/00-setup.md) | Build the crate and the CLI; know which crate owns which stage |
| 01 | [01-build-and-run.md](lessons/01-build-and-run.md) | Build a `TypedModel` node by node and run it |
| 02 | [02-stages.md](lessons/02-stages.md) | Read a graph after each stage; explain every node that appeared or vanished |
| 03 | [03-same-model-via-cli.md](lessons/03-same-model-via-cli.md) | Drive the same stages from the CLI; diff two stages numerically |
| 04 | [04-pass-lists.md](lessons/04-pass-lists.md) | Attribute each change to a named pass; bisect a pass list |
| 05 | [05-your-own-rule.md](lessons/05-your-own-rule.md) | Write a `Rewriter` rule that returns a `TypedModelPatch` |
| 06 | [06-symbolic-shapes.md](lessons/06-symbolic-shapes.md) | Build a model with a symbolic dim; know what it costs the optimiser |
| 07 | [07-metal-dispatch.md](lessons/07-metal-dispatch.md) | See how a backend swaps ops and where the host/device boundary lands |
| 08 | [08-first-contribution.md](lessons/08-first-contribution.md) | Ship a small fix following the repo's rules |

Lessons 01–04 are a single arc on one model and are best done in order.
05, 06 and 07 are independent and can be taken in any order afterwards.

## Setup

```sh
cargo build -p tract-learn
cargo build -p tract-cli --no-default-features --features onnx,pulse,extra
```

The second line is a trimmed CLI (no TensorFlow, TFLite, CUDA or transformers)
that builds much faster and is enough for every lesson here. Metal is *not* a
cargo feature — `tract-metal` is an unconditional dependency on macOS, so
Lesson 07 works with this build too.

Throughout, `tract` means `./target/debug/tract`. Run everything from the repo
root, not from `learn/`.

## Checking your work

```sh
cargo test -p tract-learn
```

Every op histogram, node count and numeric result quoted in the lessons is
asserted in [`tests/lessons.rs`](tests/lessons.rs). If a change to tract
invalidates a lesson, that test fails instead of the prose silently going stale
— so if a lesson's numbers don't match what you see, run the tests first: they
tell you whether the lesson is wrong or your command was.

## A note on imports

The lesson code uses `tract-core` and `tract-nnef` directly, not the public
`api/rs` crate. That is deliberate and specific to this course: `api/rs`
collapses the pipeline on purpose (`load` already declutters, `into_runnable`
hides optimize behind the plan), so it cannot show you an intermediate graph.
Real client code should still use `api/rs` only — see `doc/intro.md`
§"Public API".

## Progress

- [ ] 00 Setup and the crate map
- [ ] 01 Build a model by hand and run it
- [ ] 02 The stages, one at a time
- [ ] 03 The same model through the CLI
- [ ] 04 Reading the pass lists
- [ ] 05 Writing your own rule
- [ ] 06 Symbolic shapes
- [ ] 07 Metal dispatch
- [ ] 08 Your first contribution
