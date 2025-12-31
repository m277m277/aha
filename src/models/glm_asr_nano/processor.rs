use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};

use crate::{
    models::glm_asr_nano::config::GlmAsrNanoProcessorConfig,
    tokenizer::TokenizerModel,
    utils::{audio_utils::extract_audios, extract_user_text},
};

pub struct WhisperFeatureExtractor {
    feature_size: usize,
    sampling_rate: usize,
    padding_value: f32,
    hop_length: usize,
    chunk_length: usize,
    n_fft: usize,
    dither: f32,
}

pub struct GlmAsrNanoProcessor {
    sampling_rate: usize,
    chunk_length: usize,
    audio_token: String,
    audio_token_id: u32,
    max_audio_len: usize,
    default_transcription_prompt: String,
    device: Device,
}

impl GlmAsrNanoProcessor {
    pub fn new(path: &str, device: &Device, dtype: DType) -> Result<Self> {
        let path = path.to_string();
        assert!(
            std::path::Path::new(&path).exists(),
            "model path file not exists"
        );
        let processor_config_path = path.to_string() + "/processor_config.json";
        assert!(
            std::path::Path::new(&processor_config_path).exists(),
            "processor_config.json not exists in model path"
        );
        let processor_cfg: GlmAsrNanoProcessorConfig =
            serde_json::from_slice(&std::fs::read(processor_config_path)?)?;
        let audio_token = processor_cfg.audio_token.clone();
        let audio_token_id = 59260u32;
        let max_audio_len = processor_cfg.max_audio_len;
        let default_transcription_prompt = processor_cfg.default_transcription_prompt.clone();
        let sampling_rate = processor_cfg.feature_extractor.sampling_rate;
        let chunk_length = processor_cfg.feature_extractor.chunk_length;
        Ok(Self {
            sampling_rate,
            chunk_length,
            audio_token,
            audio_token_id,
            max_audio_len,
            default_transcription_prompt,
            device: device.clone(),
        })
    }

    // pub fn process_audio(&self, audios: Vec<Tensor>) -> Result<Tensor> {
    //     let window_size = self.sampling_rate * self.chunk_length;
    //     let max_windows = self.max_audio_len / self.chunk_length;
    //     let mut per_sample_windows = vec![];
    //     let mut flat_chunks = vec![];
    //     for audio_el in audios {
    //         let n_samples = audio_el.dim(0)?;
    //         let n_win = ((n_samples + window_size - 1) / window_size).max(1);
    //         let n_win = if n_win > max_windows {
    //             max_windows
    //         } else {
    //             n_win
    //         };
    //         per_sample_windows.push(n_win);
    //         let time_cap = (n_win * window_size).min(n_samples);
    //         for i in 0..n_win {
    //             let start = i * window_size;
    //             let end = ((i + 1) * window_size).min(time_cap);
    //             flat_chunks.push(audio_el.i(start..end)?);
    //         }
    //     }
    // }

    pub fn process_info(
        &self,
        mes: &ChatCompletionParameters
    ) -> Result<Tensor> {
        let audio_tensors = extract_audios(mes, &self.device, Some(self.sampling_rate))?;
        println!("audio: {}", audio_tensors[0]);
        // let audio = self.process_audio(audio_tensors)?;
        Ok(audio_tensors[0].clone())
    }
}
