use tract_core::internal::*;
use tract_learn::*;

fn main() -> TractResult<()> {
    for volume in [BiasVolume::AboveEagerFoldLimit, BiasVolume::BelowEagerFoldLimit] {
        println!("\n############ bias_volume = {volume:?} ############");

        let typed = tiny_model(volume)?;
        show_stage("after wiring (pre-declutter)", &typed);

        let decluttered = typed.clone().into_decluttered()?;
        show_stage("after declutter()", &decluttered);

        let optimized = decluttered.clone().into_optimized()?;
        show_stage("after optimize()", &optimized);

        let a = run(&typed)?;
        let b = run(&decluttered)?;
        let c = run(&optimized)?;
        println!("\npre-declutter output: {a:?}");
        println!("declutter == pre-declutter: {:?}", b.close_enough(&a, false).is_ok());
        println!("optimize  == pre-declutter: {:?}", c.close_enough(&a, false).is_ok());
    }
    Ok(())
}
