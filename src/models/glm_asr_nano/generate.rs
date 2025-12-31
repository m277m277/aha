use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::Result;
use candle_core::{DType, Device};

use crate::{
    chat_template::ChatTemplate,
    models::glm_asr_nano::{config::GlmAsrNanoProcessorConfig, processor::GlmAsrNanoProcessor},
    tokenizer::TokenizerModel,
    utils::{get_device, get_dtype},
};

pub struct GlmAsrNanoGenerateModel<'a> {
    chat_template: ChatTemplate<'a>,
    tokenizer: TokenizerModel,
    processor: GlmAsrNanoProcessor,
    // glm_asr_nano: GlmAsrNanoModel,
    device: Device,
    // eos_token_id1: u32,
    // eos_token_id2: u32,
    // eos_token_id3: u32,
    // generation_config: GlmAsrNanoGenerationConfig,
    model_name: String,
}

impl<'a> GlmAsrNanoGenerateModel<'a> {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let device = get_device(device);
        let processor = GlmAsrNanoProcessor::new(path, &device, DType::F32)?;
        // let cfg_dtype = cfg.dtype.as_str();
        // let dtype = get_dtype(dtype, cfg_dtype);
        Ok(Self {
            chat_template,
            tokenizer,
            processor,
            device,
            // eos_token_id1,
            // eos_token_id2,
            // eos_token_id3,
            model_name: "glm-asr-nano".to_string(),
        })
    }

    pub fn generate(&self, mes: ChatCompletionParameters) -> Result<()> {
        let render_text = self.chat_template.apply_chat_template(&mes)?;
        let audio = self.processor.process_info(&mes)?;
        Ok(())
    }
}
