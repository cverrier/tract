use tract_core::internal::*;
use tract_core::transform::ModelTransform;
use tract_learn::*;

fn main() -> TractResult<()> {
    let model = two_adds_model()?;
    show_stage("two_adds_model, after wiring", &model);

    let decluttered = model.clone().into_decluttered()?;
    show_stage("after stock declutter() — does tract already fold this?", &decluttered);

    let mut hoisted = decluttered.clone();
    HoistConstantAdds.transform(&mut hoisted)?;
    show_stage("after HoistConstantAdds", &hoisted);

    let before = run(&decluttered)?;
    let after = run(&hoisted)?;
    println!("\nnumerics unchanged by the rule: {}", after.close_enough(&before, false).is_ok());
    println!("output: {after:?}");
    Ok(())
}
