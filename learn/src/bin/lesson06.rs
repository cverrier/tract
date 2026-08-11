use tract_core::internal::*;
use tract_learn::*;

fn main() -> TractResult<()> {
    let symbolic = symbolic_model()?;
    show_stage("symbolic, after wiring", &symbolic);

    let decluttered = symbolic.clone().into_decluttered()?;
    show_stage("symbolic, after declutter()", &decluttered);

    let optimized = decluttered.clone().into_optimized()?;
    show_stage("symbolic, after optimize()", &optimized);

    let b = optimized.symbols.sym("B");
    let bound = decluttered.set_symbols(&[(b, M.to_dim())].into_iter().collect())?;
    show_stage("symbolic, decluttered then B bound to 16", &bound);

    let bound_optimized = bound.clone().into_optimized()?;
    show_stage("symbolic, B bound then optimize()", &bound_optimized);

    let concrete = tiny_model(BiasVolume::BelowEagerFoldLimit)?.into_optimized()?;
    println!(
        "\nbound-then-optimized matches the born-concrete model: {}",
        run(&bound_optimized)?.close_enough(&run(&concrete)?, false).is_ok()
    );
    Ok(())
}
