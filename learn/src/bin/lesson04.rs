use tract_core::internal::*;
use tract_core::optim::Optimizer;
use tract_learn::*;

fn main() -> TractResult<()> {
    println!("declutter, one patch at a time (the --declutter-step N equivalent):\n");
    for steps in 0..=4 {
        let mut model = tiny_model(BiasVolume::AboveEagerFoldLimit)?;
        Optimizer::declutter().stopping_at(steps).session().optimize(&mut model)?;
        println!("  stopping_at({steps}) -> {:?}", op_histogram(&model));
    }

    println!("\nand the same for optimize()'s codegen list:\n");
    let decluttered = tiny_model(BiasVolume::AboveEagerFoldLimit)?.into_decluttered()?;
    for steps in 0..=4 {
        let mut model = decluttered.clone();
        Optimizer::codegen().stopping_at(steps).session().optimize(&mut model)?;
        println!("  stopping_at({steps}) -> {:?}", op_histogram(&model));
    }
    Ok(())
}
