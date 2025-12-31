use serde::{Deserialize};

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct GlmAsrNanoProcessorConfig {
    pub audio_token: String,
    pub default_transcription_prompt: String,
    pub feature_extractor: FeatureExtractor,
    pub max_audio_len: usize,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct FeatureExtractor {
    pub chunk_length: usize,
    pub dither: f32,
    pub feature_size: usize,
    pub hop_length: usize,
    pub n_fft: usize,
    pub n_samples: usize,
    pub nb_max_frames: usize,
    pub padding_side: String,
    pub padding_value: f32,
    pub return_attention_mask: bool,
    pub sampling_rate: usize,
}