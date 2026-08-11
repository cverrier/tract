# 00 — Setup and the crate map

## Read

- `AGENTS.md` §"Crate map" — the dependency table. Two minutes.
- `doc/pipeline.md` §"The stages" — the table of stage → method → crate.

## Build

```sh
cargo build -p tract-learn
cargo build -p tract-cli --no-default-features --features onnx,pulse,extra
cargo test -p tract-learn
```

All three should succeed. `cargo test` runs 9 tests; they are the executable
version of everything the later lessons claim.

## Predict

Before running anything, answer from the crate map alone:

1. Which crate owns `TypedModel`?
2. Which crate owns `InferenceModel`, and why is it a *different* crate?
3. `tract-transformers` has no direct dependency on `tract-core`. How does it
   get at core types?
4. You want to know whether `into_optimized` is public API. Which file settles
   it?

<details>
<summary>Answers</summary>

1. `core`. It also owns the op trait, the passes, the rewriter and
   `TypedModelPatch`.
2. `hir` — the "high-level intermediate representation", the untyped graph that
   exists *before* type analysis. It depends on `core`, not the reverse: the
   typed world knows nothing about the inference world. That direction is why
   `into_typed()` lives in `hir` and is the only bridge.
3. Via `tract_nnef::tract_core` (`AGENTS.md` crate-map footnote). It keeps the
   dependency graph shallow.
4. `api/rs/src/lib.rs`, and nothing else. An item being `pub` in `core` says
   nothing about it being public API. Spoiler for later lessons:
   `into_optimized` is *not* on that surface.

</details>

## Run

Check which runtimes your binary actually has:

```sh
./target/debug/tract list-runtimes
```

On an Apple-silicon Mac:

```
 * cpu
 * metal
 * unoptimized
```

Three things worth noticing now, because they explain flags you will meet later:

- **`cpu`** is `DefaultRuntime`. `runtime_for_name` also answers to `default`
  as an alias for it.
- **`unoptimized`** exists *only in the CLI* (`cli/src/runtimes.rs`). It plans
  the graph without optimising, which is what lets `tract` run and inspect an
  intermediate stage. It is also the default when you forget `-O`.
- **`metal`** is here without any cargo feature: `tract-metal` is an
  unconditional dependency of `tract-cli` on macOS/iOS
  (`cli/Cargo.toml`). On Linux you would see `cuda` here instead, subject to a
  feature and a toolchain.

## Explain

The shape to carry into Lesson 01, from `doc/pipeline.md`:

```
load ──► analyse ──► incorporate ──► into_typed ──► declutter ──► optimize ──► plan ──► run
         └────────── tract-hir ──────────────────┘  └──────── tract-core ──────────────┘
                    InferenceModel                            TypedModel
```

Two things this course will keep coming back to:

- **`declutter` is target-independent, `optimize` is not.** Decluttered is the
  portable form you serialise to NNEF. Optimised is only valid for the machine
  that produced it. Lesson 03 makes that concrete by watching serialisation
  fail.
- **The second half of the pipeline belongs to a `Runtime`.** `optimize → plan`
  is what `Runtime::prepare` does, and each backend does it differently. Lesson
  07 is that difference.

There is no `compile()` anywhere. If you go looking for one you will waste ten
minutes — the pipeline is a chain of separate methods precisely so you can stop
between any two of them, which is what makes this course possible.

---

Next: [01 — Build a model by hand and run it](01-build-and-run.md)
