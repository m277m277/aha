use std::time::Instant;

use aha::models::{GenerateModel, qwen3vl::generate::Qwen3VLGenerateModel};
use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::Result;

#[test]
fn gelab_zero_generate() -> Result<()> {
    // test with cuda: RUST_BACKTRACE=1 cargo test -F cuda gelab_zero_generate -r -- --nocapture

    let model_path = "/home/jhq/huggingface_model/stepfun-ai/GELab-Zero-4B-preview";

    let message = r#"
    {
        "model": "gelab-zero",
        "messages": [
            {
                "role": "user",
                "content": [    
                    {
                        "type": "text", 
                        "text": "Hello, GELab-Zero!"
                    }
                ]
            }
        ]
    }
    "#;
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut qwen3vl = Qwen3VLGenerateModel::init(model_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);

    let i_start = Instant::now();
    let res = qwen3vl.generate(mes)?;
    let i_duration = i_start.elapsed();
    println!("generate: \n {:?}", res);
    println!("Time elapsed in generate is: {:?}", i_duration);
    Ok(())
}
