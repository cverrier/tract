# 08 — Your first contribution

## Read

- `CLAUDE.md` at the repo root — the contributor rules. All of it; it is short.
- `AGENTS.md` §"Commit hygiene", §"Style", §"Things to avoid".

## The point of this lesson

The four bugs below are real, small, and were found *by doing lessons 03 and 04*.
That is the actual skill this course was for: reading a pass list or a flag
definition closely enough to notice when it doesn't do what it says.

They are left unfixed on purpose. Pick one.

## Four real defects, verified

### 1. `--pass optimize` does not optimize

```sh
T=./target/debug/tract
$T learn/models/tiny --pass optimize dump --audit-json   # Add, EinSum  — unoptimised
$T -O learn/models/tiny dump --audit-json                # OptMatMul    — optimised
```

`cli/src/params.rs` sets `stop_at = "optimize"`, but there is no
`stage!("optimize", …)` in `load_and_declutter`. Optimisation is decided
separately by runtime selection, which keys off the `-O` flag and never looks at
`--pass`. So the value is accepted, validated by clap, and silently does nothing.

*Design question to settle before coding:* should `--pass optimize` imply the
default runtime, or should it be rejected as not-a-stage? Both are defensible.
Argue for one in the PR rather than picking silently.

### 2. Two `STAGES` entries are unreachable

`cli/src/main.rs` advertises `nnef-cycle-declutter` and `tflite-cycle-declutter`.
The actual `stage!` invocations in `cli/src/params.rs` are named
`"nnef-declutter"` and `"tflite-declutter"`. So:

- `--pass nnef-cycle-declutter` passes clap validation, matches no stage, and runs
  to the end — a silent no-op.
- `--pass nnef-declutter`, the name that would work, is rejected by clap.

The smallest honest fix is one of the two names; check which spelling anything
else depends on before choosing.

### 3. `--optimize-step` is a dead flag

Declared in `cli/src/main.rs` as *"Stop optimizing process after application of
patch number N"*. Grep the repo: that declaration is the only occurrence.
Meanwhile `--declutter-step` and `--declutter-set-step` are wired up in
`cli/src/params.rs`.

You already know from Lesson 04 that stepping the codegen list is genuinely
useful — `Optimizer::codegen().stopping_at(n)` is exactly what the flag promises,
and `stopping_at` is public. So this is a small implementation, not just a
deletion. It is the most self-contained of the four.

### 4. `doc/cli-recipe.md` is stale about `--nnef-tract-core`

The doc tells you to pass `--nnef-tract-core` when loading NNEF. In
`cli/src/main.rs` that flag is declared as `"no-op, kept for backward
compatibility"` and hidden from `--help`; the `tract_core` registry is enabled by
default and the live flag is `--no-nnef-tract-core`. Every `runme.sh` under
`harness/nnef-test-cases/` still passes the no-op.

A docs-only fix, and the gentlest of the four — but read `doc/README.md`'s own
advice first: *"If something here disagrees with the source, trust the source —
and consider patching the doc."*

## The workflow

```sh
git checkout -b fix/<short-description> main    # branch off main, not off learn

# ... make the change ...

cargo fmt --all                                 # repo-wide, never per-crate
cargo clippy --workspace                        # must be clean
cargo test -p tract-cli                         # plus whatever you touched
```

`rust-toolchain.toml` pins stable, so bare `cargo fmt` is already the version CI
checks against. Don't override the toolchain.

### Testing a CLI fix

From `CLAUDE.md`: add synthetic cases under `harness/nnef-test-cases/`, driven by
a `runme.sh`, **not** as new Rust integration tests. If the assertion you need
isn't expressible through the CLI, extend the CLI.

A case directory is a `graph.nnef` plus a `runme.sh` with this preamble:

```sh
#!/bin/sh
cd `dirname $0`
set -ex
: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}
```

`.travis/cli-tests.sh` finds every `runme.sh` automatically, so a new directory is
picked up with no registration step. Good models to copy:

- `harness/nnef-test-cases/copy-identity/` — the minimum, 120 bytes of NNEF.
- `harness/nnef-test-cases/sign-abs-integers/` — uses `--assert-op-count` to pin
  *which* code path ran, not just that the numbers came out right. Its comments
  explain why each assertion exists. This is the standard to aim for.

For bug 1 or 2, `--assert-op-count EinSum 0` after `--pass optimize` would be the
assertion that fails today and passes after your fix.

### Commit message

One short paragraph: what was wrong, and the fix. Nothing else. No consequence
chains, no `Result:`/`Symptom:` sections, no bullet list of every place the bug
showed up. For bug 3:

> Wire `--optimize-step` through to the codegen optimizer. The flag was declared
> but never read, so stepping the optimize pass list was impossible; it now sets
> `Optimizer::stopping_at` the same way `--declutter-step` does.

### Comments

Default to none — names carry the meaning, and a comment signals a hidden
constraint or invariant. Do add a `///` doc comment on anything public or
non-trivial, describing the *current contract*. Never narrate history ("used to
be", "previously") in either kind.

### PR

Open with one or two sentences: what and why. Then let a human handle the review
conversation — the maintainer wants to talk to the author.

## Where to go after this

- Add a case to a `suite-*` crate — the standard way to test an op, and the most
  common shape of a tract PR.
- Read `doc/op.md` and implement a toy op end to end: `output_facts`, `eval`,
  `declutter`, and NNEF ser/de.
- Read `doc/kernel-notes.md` and `doc/cost-model.md` if the linalg layer is what
  interests you; `tract hwbench` is the tool.
- Steer clear of `pulse`/`pulse-opl` until you have more mileage. The streaming
  invariants are subtle and the repo says so explicitly.

---

Back to the [course index](../README.md).
