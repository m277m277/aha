use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::Result;
use candle_core::{DType, Device};

use crate::{
    chat_template::ChatTemplate,
    models::{
        lfm2::config::Lfm2GenerateConfig,
        lfm2vl::{config::Lfm2VLConfig, processor::Lfm2VLProcessor},
    },
    tokenizer::TokenizerModel,
    utils::{find_type_files, get_device, get_dtype, get_logit_processor},
};

pub struct Lfm2VLGenerateModel<'a> {
    chat_template: ChatTemplate<'a>,
    tokenizer: TokenizerModel,
    device: Device,
    // model: Lfm2VLModel,
    processor: Lfm2VLProcessor,
    eos_token_id: u32,
    model_name: String,
}
impl<'a> Lfm2VLGenerateModel<'a> {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let device = get_device(device);
        let gen_cfg_path = path.to_string() + "/generation_config.json";
        let gen_cfg: Lfm2GenerateConfig = serde_json::from_slice(&std::fs::read(gen_cfg_path)?)?;
        let cfg_path = path.to_string() + "/config.json";
        let cfg: Lfm2VLConfig = serde_json::from_slice(&std::fs::read(cfg_path)?)?;

        let model_path = find_type_files(path, "safetensors")?;
        let dtype = get_dtype(dtype, &cfg.dtype);
        // let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_path, dtype, &device)? };
        // let model = Lfm2Model::new(vb, &cfg)?;
        let processor = Lfm2VLProcessor::new(path, dtype, &device)?;
        let eos_token_id = gen_cfg.eos_token_id;
        let model_name = std::path::Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("lfm2.5-vl")
            .to_string();
        Ok(Self {
            chat_template,
            tokenizer,
            device,
            // model,
            processor,
            eos_token_id,
            model_name,
        })
    }

    pub fn generate(&mut self, mes: ChatCompletionParameters) -> Result<()> {
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let mut logits = get_logit_processor(
            mes.temperature,
            mes.top_p,
            None,
            mes.seed.unwrap_or(34562) as u64,
        );
        let (pixel_values, pixel_attention_mask, spatial_shapes, text) =
            self.processor.process_info(&mes, &mes_render)?;
        let input_ids = self.tokenizer.text_encode(text, &self.device)?;
        println!("pixel_values: {}", pixel_values);
        println!("pixel_attention_mask: {}", pixel_attention_mask);
        println!("spatial_shapes: {}", spatial_shapes);
        println!("input_ids: {}", input_ids);
        Ok(())
    }
}
