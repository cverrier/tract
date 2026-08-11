use std::path::Path;

use tract_core::internal::*;
use tract_learn::*;

fn main() -> TractResult<()> {
    let dir = Path::new("learn/models/tiny");
    let decluttered = tiny_model(BiasVolume::AboveEagerFoldLimit)?.into_decluttered()?;
    write_nnef(&decluttered, dir)?;

    println!("wrote {}", dir.display());
    for entry in std::fs::read_dir(dir)? {
        println!("  {}", entry?.file_name().to_string_lossy());
    }
    println!("\n--- graph.nnef ---");
    println!("{}", std::fs::read_to_string(dir.join("graph.nnef"))?);
    println!("Rust-side histogram at this stage: {:?}", op_histogram(&decluttered));

    let optimized = decluttered.into_optimized()?;
    match write_nnef(&optimized, Path::new("learn/models/tiny-optimized")) {
        Ok(()) => println!("\nunexpected: the optimized model serialized"),
        Err(e) => println!("\nserializing the OPTIMIZED model fails, as it must:\n  {e:#}"),
    }
    Ok(())
}
