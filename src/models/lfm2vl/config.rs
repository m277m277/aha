use candle_nn::Activation;

use crate::models::lfm2::config::Lfm2Config;


#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Lfm2VLConfig {
    pub do_image_splitting: bool,
    pub downsample_factor: usize,
    pub dtype: String,
    pub encoder_patch_size: usize,
    pub image_token_id: u32,
    pub max_image_tokens: usize,
    pub max_pixels_tolerance: f64,
    pub max_tiles: usize,
    pub min_image_tokens: usize,
    pub min_tiles: usize,
    pub model_type: String,
    pub projector_bias: bool,
    pub projector_hidden_act: Activation,
    pub projector_hidden_size: usize,
    pub projector_use_layernorm: bool,
    pub text_config: Lfm2Config,
    pub tile_size: usize,
    pub use_image_special_tokens: bool,
    pub use_thumbnail: bool,
    pub vision_config: Lfm2VLVisionConfig,
}


#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Lfm2VLVisionConfig {
    pub attention_dropout: f64,
    pub dtype: String,
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub layer_norm_eps: f64,
    pub model_type: String,
    pub num_attention_heads: u32,
    pub num_channels: u32,
    pub num_hidden_layers: usize,
    pub num_patches: usize,
    pub patch_size: usize,
    pub vision_use_head: bool,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Lfm2ImageConfig {
    pub do_image_splitting: bool,
    pub do_normalize: bool,
    pub do_pad: bool,
    pub do_rescale: bool,
    pub do_resize: bool,
    pub downsample_factor: usize,
    pub encoder_patch_size: usize,
    pub image_mean: Vec<f64>,
    pub image_std: Vec<f64>,
    pub max_image_tokens: usize,
    pub max_num_patches: usize,
    pub max_pixels_tolerance: f64,
    pub max_tiles: usize,
    pub min_image_tokens: usize,
    pub min_tiles: usize,
    pub resample: usize,
    pub rescale_factor: f64,
    pub size: Size,
    pub tile_size: usize,
    pub use_thumbnail: bool,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Size {
    pub height: usize,
    pub width: usize,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Lfm2ProcessorConfig {
    pub image_processor: Lfm2ImageConfig,
}

