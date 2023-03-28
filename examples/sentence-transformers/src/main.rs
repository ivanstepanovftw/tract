use tokenizers::tokenizer::Tokenizer;
use tract_onnx::prelude::*;

fn main() -> TractResult<()> {
    let question = "That is a happy person".to_string();

    let tokenizer = Tokenizer::from_bytes(include_bytes!("../onnx/tokenizer.json")).unwrap();

    // Tokenize question
    let input = tokenizer.encode(question, false).unwrap();
    let input_ids = input.get_ids();
    let attention_mask = input.get_attention_mask();
    let token_type_ids = input.get_type_ids();
    let length = input_ids.len();

    let input_ids: Tensor = tract_ndarray::Array2::from_shape_vec(
        (1, length),
        input_ids.iter().map(|&x| x as i64).collect(),
    )?.into();
    let attention_mask: Tensor = tract_ndarray::Array2::from_shape_vec(
        (1, length),
        attention_mask.iter().map(|&x| x as i64).collect(),
    )?.into();

    let token_type_ids: Tensor = tract_ndarray::Array2::from_shape_vec(
        (1, length),
        token_type_ids.iter().map(|&x| x as i64).collect(),
    )?.into();


    // Get model
    let t1 = std::time::Instant::now();
    let model = onnx()
    .model_for_read(&mut &include_bytes!("../onnx/model.onnx")[..])?
    .into_optimized()?
    .into_runnable()?;
    let dt = std::time::Instant::now() - t1;
    println!("model loaded in {} ms", dt.as_millis());

    // Time to run the model
    let t1 = std::time::Instant::now();
    let outputs = model.run(tvec!(input_ids.into(), attention_mask.into(), token_type_ids.into()))?;
    let dt = std::time::Instant::now() - t1;
    println!("model run in {} ms", dt.as_millis());
    println!("outputs[0]: {:?}", outputs[0]);


    Ok(())
}
