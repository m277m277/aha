use aha::models::glm_asr_nano::generate::GlmAsrNanoGenerateModel;
use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::{Result};

#[test]
fn glm_asr_nano_generate() -> Result<()> {
    // RUST_BACKTRACE=1 cargo test -F cuda glm_asr_nano_generate -r -- --nocapture
    let model_path = "/home/jhq/huggingface_model/zai-org/GLM-ASR-Nano-2512/";
    let message = r#"
    {
        "model": "glm-asr-nano",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio_url": 
                        {
                            "url": "file://./assets/audio/voice_01.wav"
                        }
                    },              
                    {
                        "type": "text", 
                        "text": "Please transcribe this audio into text"
                    }
                ]
            }
        ]
    }
    "#;
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let glm_asr_model = GlmAsrNanoGenerateModel::init(model_path, None, None)?;
    let _ = glm_asr_model.generate(mes)?;
    Ok(())
}