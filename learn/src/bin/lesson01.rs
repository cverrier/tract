use tract_core::internal::*;
use tract_learn::*;

fn main() -> TractResult<()> {
    let model = tiny_model(BiasVolume::AboveEagerFoldLimit)?;

    println!("{model}");
    println!("input  fact: {:?}", model.input_fact(0)?);
    println!("output fact: {:?}", model.output_fact(0)?);

    let output = run(&model)?;
    println!("\ninput:  {:?}", sample_input()?);
    println!("output: {output:?}");

    let row0: f32 = (0..K).map(|c| (c as f32 / 10.0) + 1.0).sum::<f32>() * 0.5;
    println!("\nrow 0 by hand: sum over k of (input[0,k] + 1) * 0.5 = {row0}");
    Ok(())
}
