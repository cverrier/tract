# 06 — Symbolic shapes

## Read

- `doc/symbolic-shapes.md` §"What `TDim` is" and §"Variants worth a closer look".
  The `Broadcast` vs `Max` distinction and the `#` operator in the textual syntax
  are the two things people get wrong.

## Predict

`symbolic_model()` is the Lesson 01 model with the row count replaced by a symbol
`B`. `K = 8` and `N = 4` stay concrete.

1. Does `declutter()` still delete the neutral `Mul` and fold the constants?
2. Does `optimize()` still lower the `EinSum` to `OptMatMul`, with an unknown
   number of rows?
3. If you bind `B = 16` *after* decluttering and then optimise, do you get the
   same graph as the born-concrete model?

## Run

```sh
cargo run -p tract-learn --bin lesson06
```

### Wiring, then declutter

```
0 | Source  input  => B,8,F32
...
=== symbolic, after declutter() ===
op histogram: {"Add": 1, "Const": 2, "EinSum": 1, "Source": 1}
nodes: 5
```

Prediction 1: **yes, unchanged.** Both rules were about *values*, not shapes —
`Mul` by a uniform 1 is a no-op whatever the row count, and folding two constants
needs no knowledge of `B`. Every fact in the graph now carries `B` as a `TDim`
where a `usize` used to be, and the passes are written against that algebra
throughout.

### Optimize

```
=== symbolic, after optimize() ===
2 | Add            add            => B,8,F32
3 | OptMatMulPack  matmul.pack_a  => ,F32 🔍 DynPackedExoticFact { k: Val(8), mn: Sym(B), packers: [PackedF32[32]@128+1] }
4 | OptMatMul      matmul         => B,4,F32

op histogram: {"Add": 1, "Const": 1, "OptMatMul": 1, "OptMatMulPack": 1, "Source": 1}
```

Prediction 2: **it lowers.** A symbolic row count does not block codegen. Read
the exotic fact: `k: Val(8)` is known, `mn: Sym(B)` is not, and the graph happily
carries a `Sym` into the packing layout.

### Careful with the comparison

`Add` is a plain `Add` here, where Lesson 02's optimised graph showed
`OptAddUnicast`. It is tempting — and wrong — to blame the symbol. This model's
constant addend is `[1, 8]`; Lesson 02's was `[16, 8]`. Compare like with like:

```
concrete, [1, 8]  bias -> {"Add": 1, ..., "OptMatMul": 1, "OptMatMulPack": 1}
concrete, [16, 8] bias -> {"OptAddUnicast": 1, ..., "OptMatMul": 1, "OptMatMulPack": 1}
symbolic, [1, 8]  bias -> {"Add": 1, ..., "OptMatMul": 1, "OptMatMulPack": 1}
```

The symbolic model matches the concrete `[1, 8]` model exactly. The `Add`/
`OptAddUnicast` difference is about **shape, not symbols**: the unicast codegen
counts the *matching trailing dimensions* of the two operands and needs at least
32 elements (`core/src/ops/binary.rs`). `[16,8]` against `[16,8]` gives
`16*8 = 128` and fires; `[16,8]` against `[1,8]` stops at the first mismatched
axis and gives `8`, which declines. Adding `model.symbols.add_assertion("B >= 64")`
does not change the symbolic result, which confirms it: no predicate about `B` was
ever the obstacle.

Worth doing this check every time you attribute a missing optimisation to a
symbolic dim. Two graphs that differ in two ways tell you nothing about either.

### What the symbol actually costs

One real difference survives the controlled comparison — the packer:

```
symbolic     mn: Sym(B)   packers: [PackedF32[32]@128+1]
B bound = 16 mn: Val(16)  packers: [PackedF32[8]@16+1]
```

With the row count unknown, tract cannot size the packing to it and picks a
conservative layout. The kernel still runs and the answers are identical; you are
paying in memory traffic, not correctness.

### Bind, then optimise

```
=== symbolic, B bound then optimize() ===
3 | OptMatMulPack  matmul.pack_a  => ... { k: Val(8), mn: Val(16), packers: [PackedF32[8]@16+1] }

bound-then-optimized matches the born-concrete model: true
```

Prediction 3: yes. `set_symbols` substitutes `B = 16` into every fact, and from
there codegen has the same information it would have had all along. The packer is
back to `PackedF32[8]@16+1`.

The lesson in the ordering: **bind as early as you can.** A model optimised with
`B` symbolic is correct, but codegen had to hedge. If you know the batch size at
load time, say so at load time — and if you only know a bound, say *that*, with
`add_assertion`, so the simplifier can discharge predicates that would otherwise
block a rewrite.

## The API

```rust
let mut model = TypedModel::default();
let b = model.symbols.sym("B");
let input = model.add_source("input", f32::fact(dims!(b.clone(), K)))?;
```

- `model.symbols` is the graph's `SymbolScope`. `sym("B")` interns a name and
  returns the same `Symbol` every time.
- `dims!(...)` builds a `Vec<TDim>` from a mix of symbols and integers. Plain
  `f32::fact([M, K])` only takes concrete dims.
- To bind: `model.set_symbols(&HashMap<Symbol, TDim>)` returns a new model.
- Constraints: `model.symbols.add_assertion("B >= 1")?`. These let the simplifier
  discharge predicates it otherwise couldn't — the direct cure for "why didn't my
  optimisation fire".

## On the CLI

```sh
T=./target/debug/tract

$T -i B,8,f32 model.onnx dump              # declare a symbolic input dim
$T -i B,8,f32 --set B=16 model.onnx dump   # bind it, graph-wide
$T -i B,8,f32 --assert "B>0" model.onnx dump
```

Two syntax notes that cost people time:

- Shape specs are **comma**-separated, including the dtype: `1,3,224,224,f32`.
  `1x3x224x224` is not accepted anywhere. `_` means "unknown".
- `--set` exists twice with different meanings. As a *root* flag it rewrites the
  graph (there are `set` and `set-declutter` pipeline stages for it). On the `run`
  subcommand it only binds symbols for that execution. Root flags go before the
  subcommand.
- `--hint B=16` is neither: it gives the planner a typical value for memory
  sizing without touching the graph.

## Exercise

Make `K` — the *reduction* dimension — symbolic instead of `B`, with non-constant
weights so the shape stays legal, and re-run. Predict first: `K` is the dimension
the micro-kernel loops over, so does the `EinSum` still lower?

<details>
<summary>Answer</summary>

It still lowers: `{"OptMatMul": 1, "OptMatMulPack": 2, "Source": 2}`, identical to
the same graph with `K = 8` concrete. `OptMatMul` carries `k` as a `TDim` and
resolves it at run time.

So on this graph, no symbolic dimension blocks lowering at all — the honest
summary of this lesson is that tract's algebra is doing more work than you
probably expected, and the cost of a symbol shows up in kernel and layout
*choices* rather than in whether an optimisation happens.

</details>

Harder: find a rewrite that a symbol genuinely blocks. `Reduce`, `Slice` and the
`ChangeAxes` pass are better hunting grounds than matmul, because they need to
compare dims rather than just carry them. Use `--pass declutter` and
`stopping_at` from Lesson 04 to see which patch stops firing.

---

Next: [07 — Metal dispatch](07-metal-dispatch.md)
