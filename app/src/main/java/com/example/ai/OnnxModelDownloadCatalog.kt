package com.example.ai

data class OnnxDownloadLinkItem(
    val id: String,
    val titleEn: String,
    val titleTa: String,
    val subtitleEn: String,
    val subtitleTa: String,
    val sizeBadge: String,
    val url: String,
    val isDirectFile: Boolean = false
)

data class OnnxDownloadGroup(
    val id: String,
    val badgeLabelEn: String,
    val badgeLabelTa: String,
    val titleEn: String,
    val titleTa: String,
    val descriptionEn: String,
    val descriptionTa: String,
    val isRecommended: Boolean,
    val links: List<OnnxDownloadLinkItem>
)

object OnnxModelDownloadCatalog {

    val groups: List<OnnxDownloadGroup> = listOf(
        OnnxDownloadGroup(
            id = "lcm_fast_mobile",
            badgeLabelEn = "RECOMMENDED • FAST MOBILE",
            badgeLabelTa = "பரிந்துரைக்கப்படுகிறது • வேகமான மொபைல் மாடல்",
            titleEn = "Option 1: Fast LCM INT8 UNet (~860 MB)",
            titleTa = "விருப்பம் 1: வேகமான LCM INT8 UNet (~860 MB)",
            descriptionEn = "Best for Android phones (4–6 steps, low RAM, auto-optimized on first run). Download BOTH model.onnx and model_int8.onnx.data into a 'unet' subfolder without renaming.",
            descriptionTa = "மொபைலில் குறைந்த RAM-ல் 4–6 படிகளில் வேகமாகப் படம் உருவாக்க சிறந்தது (முதல் முறை தானாகவே Android CPU-க்கு optimize ஆகும்). model.onnx மற்றும் model_int8.onnx.data இரண்டையும் பெயர் மாற்றாமல் 'unet' ஃபோல்டரில் சேமிக்கவும்.",
            isRecommended = true,
            links = listOf(
                OnnxDownloadLinkItem(
                    id = "lcm_unet_folder",
                    titleEn = "UNet Folder Page (Hugging Face)",
                    titleTa = "UNet மாடல் பக்கம் (Hugging Face)",
                    subtitleEn = "Browse model.onnx + model_int8.onnx.data in browser",
                    subtitleTa = "model.onnx + model_int8.onnx.data ஃபைல்களைப் பார்க்க",
                    sizeBadge = "Folder",
                    url = "https://huggingface.co/latentdivergence/lcm-dreamshaper-int8-onnx/tree/main/unet",
                    isDirectFile = false
                ),
                OnnxDownloadLinkItem(
                    id = "lcm_unet_graph",
                    titleEn = "1. model.onnx (UNet Graph)",
                    titleTa = "1. model.onnx (UNet கட்டமைப்பு)",
                    subtitleEn = "Required graph file — keep name as model.onnx",
                    subtitleTa = "பெயரை மாற்றாமல் model.onnx என்றே சேமிக்கவும்",
                    sizeBadge = "~3.1 MB",
                    url = "https://huggingface.co/latentdivergence/lcm-dreamshaper-int8-onnx/resolve/main/unet/model.onnx",
                    isDirectFile = true
                ),
                OnnxDownloadLinkItem(
                    id = "lcm_unet_data",
                    titleEn = "2. model_int8.onnx.data (UNet Weights)",
                    titleTa = "2. model_int8.onnx.data (முக்கிய எடை ஃபைல்)",
                    subtitleEn = "Required INT8 external weights — keep exact name model_int8.onnx.data",
                    subtitleTa = "பெயரை மாற்றாமல் model_int8.onnx.data என்றே சேமிக்கவும்",
                    sizeBadge = "~861 MB",
                    url = "https://huggingface.co/latentdivergence/lcm-dreamshaper-int8-onnx/resolve/main/unet/model_int8.onnx.data",
                    isDirectFile = true
                ),
                OnnxDownloadLinkItem(
                    id = "lcm_full_repo",
                    titleEn = "Full LCM INT8 Pipeline Repo (All 4 Core Models)",
                    titleTa = "முழு LCM INT8 தொகுப்பு (4 Core ONNX மாடல்கள்)",
                    subtitleEn = "Includes text_encoder, unet, vae_decoder, vae_encoder & tokenizer",
                    subtitleTa = "text_encoder, unet, vae_decoder, vae_encoder அனைத்தும் உள்ளது",
                    sizeBadge = "Full Repo",
                    url = "https://huggingface.co/latentdivergence/lcm-dreamshaper-int8-onnx/tree/main",
                    isDirectFile = false
                ),
                OnnxDownloadLinkItem(
                    id = "lcm_fp16_repo",
                    titleEn = "Alternative: LCM Dreamshaper v7 FP16 ONNX Repo",
                    titleTa = "மாற்று விருப்பம்: LCM Dreamshaper v7 FP16 ONNX தொகுப்பு",
                    subtitleEn = "Native FP16/FP32 LCM ONNX models without INT8 quantization",
                    subtitleTa = "INT8 இல்லாத நேரடி LCM ONNX மாடல் தொகுப்பு",
                    sizeBadge = "FP16 Repo",
                    url = "https://huggingface.co/aislamov/lcm-dreamshaper-v7-onnx/tree/main",
                    isDirectFile = false
                )
            )
        ),
        OnnxDownloadGroup(
            id = "lcm_fast_mobile_fp32",
            badgeLabelEn = "ALTERNATIVE • FAST LCM FP32 (4–6 STEPS)",
            badgeLabelTa = "மாற்று வேகமான மாடல் • Fast LCM FP32 (4–6 படிகள்)",
            titleEn = "Option 2: Fast LCM FP32 UNet (~3.4 GB)",
            titleTa = "விருப்பம் 2: வேகமான Fast LCM FP32 UNet (~3.4 GB)",
            descriptionEn = "Full-precision 4–6 step Fast LCM model (no quantization). Download model.onnx and model.onnx_data together into a 'unet' folder without renaming.",
            descriptionTa = "4–6 படிகளில் வேகமாக இயங்கும் முழு அளவு Fast LCM மாடல். model.onnx மற்றும் model.onnx_data இரண்டையும் பெயர் மாற்றாமல் 'unet' ஃபோல்டரில் சேமிக்கவும்.",
            isRecommended = false,
            links = listOf(
                OnnxDownloadLinkItem(
                    id = "lcm_hex_unet_folder",
                    titleEn = "Fast LCM UNet Folder (TheyCallMeHex)",
                    titleTa = "Fast LCM UNet பக்கம் (TheyCallMeHex)",
                    subtitleEn = "Full FP32 Fast LCM UNet directory",
                    subtitleTa = "முழு அளவு Fast LCM UNet ஃபைல்கள் பக்கம்",
                    sizeBadge = "Folder",
                    url = "https://huggingface.co/TheyCallMeHex/LCM-Dreamshaper-V7-ONNX/tree/main/unet",
                    isDirectFile = false
                ),
                OnnxDownloadLinkItem(
                    id = "lcm_hex_unet_graph",
                    titleEn = "1. model.onnx (Graph Shell)",
                    titleTa = "1. model.onnx (கட்டமைப்பு ஃபைல்)",
                    subtitleEn = "Keep exact name model.onnx in unet folder",
                    subtitleTa = "பெயரை மாற்றாமல் model.onnx என்றே சேமிக்கவும்",
                    sizeBadge = "~1.9 MB",
                    url = "https://huggingface.co/TheyCallMeHex/LCM-Dreamshaper-V7-ONNX/resolve/main/unet/model.onnx",
                    isDirectFile = true
                ),
                OnnxDownloadLinkItem(
                    id = "lcm_hex_unet_data",
                    titleEn = "2. model.onnx_data (Weights)",
                    titleTa = "2. model.onnx_data (முக்கிய எடை ஃபைல்)",
                    subtitleEn = "External FP32 weights — keep exact name model.onnx_data",
                    subtitleTa = "பெயரை மாற்றாமல் model.onnx_data என்றே வைக்கவும்",
                    sizeBadge = "~3.4 GB",
                    url = "https://huggingface.co/TheyCallMeHex/LCM-Dreamshaper-V7-ONNX/resolve/main/unet/model.onnx_data",
                    isDirectFile = true
                ),
                OnnxDownloadLinkItem(
                    id = "lcm_hex_full_repo",
                    titleEn = "Full LCM FP32 Pipeline Repo (All Core Models)",
                    titleTa = "முழு LCM FP32 தொகுப்பு பக்கம் (அனைத்து மாடல்கள்)",
                    subtitleEn = "Includes text_encoder, unet, vae_decoder, vae_encoder",
                    subtitleTa = "text_encoder, unet, vae_decoder, vae_encoder அனைத்தும் உள்ளது",
                    sizeBadge = "Full Repo",
                    url = "https://huggingface.co/TheyCallMeHex/LCM-Dreamshaper-V7-ONNX/tree/main",
                    isDirectFile = false
                )
            )
        ),
        OnnxDownloadGroup(
            id = "sd15_fp32_weights_pb",
            badgeLabelEn = "STANDARD • FULL FP32",
            badgeLabelTa = "சாதாரண முழு அளவு • FP32 (3.44 GB)",
            titleEn = "Option 3: Standard SD UNet (model.onnx + weights.pb)",
            titleTa = "விருப்பம் 3: முழு FP32 UNet (model.onnx + weights.pb)",
            descriptionEn = "Matches 1.09 MB split UNet format. Requires 3.44 GB storage and high RAM. Download matching model.onnx + weights.pb pair together.",
            descriptionTa = "1.09 MB unet.onnx வகையான முழு மாடல். இதற்கு 3.44 GB இடமும் அதிக RAM-உம் தேவை. ஒரே தொகுப்பில் உள்ள model.onnx + weights.pb ஜோடியை ஒன்றாகப் பதிவிறக்கவும்.",
            isRecommended = false,
            links = listOf(
                OnnxDownloadLinkItem(
                    id = "sd15_onnx_community_unet",
                    titleEn = "SD 1.5 ONNX UNet Folder (onnx-community)",
                    titleTa = "SD 1.5 ONNX UNet பக்கம் (onnx-community)",
                    subtitleEn = "Official Stable Diffusion v1.5 ONNX UNet directory",
                    subtitleTa = "Stable Diffusion v1.5 ONNX UNet அதிகாரப்பூர்வ பக்கம்",
                    sizeBadge = "Folder",
                    url = "https://huggingface.co/onnx-community/stable-diffusion-v1-5-ONNX/tree/main/unet",
                    isDirectFile = false
                ),
                OnnxDownloadLinkItem(
                    id = "sd_compvis_unet_graph",
                    titleEn = "Paired model.onnx (1.09 MB Graph)",
                    titleTa = "இணைந்த model.onnx (1.09 MB கட்டமைப்பு)",
                    subtitleEn = "Matches weights.pb below — keep in unet folder",
                    subtitleTa = "கீழுள்ள weights.pb உடன் பொருந்தும் model.onnx",
                    sizeBadge = "1.09 MB",
                    url = "https://huggingface.co/CompVis/stable-diffusion-v1-4/resolve/onnx/unet/model.onnx",
                    isDirectFile = true
                ),
                OnnxDownloadLinkItem(
                    id = "sd_compvis_weights_pb",
                    titleEn = "Paired weights.pb (External Weights)",
                    titleTa = "இணைந்த weights.pb (3.44 GB எடை ஃபைல்)",
                    subtitleEn = "Keep exact filename weights.pb alongside model.onnx",
                    subtitleTa = "பெயரை மாற்றாமல் weights.pb என்றே model.onnx அருகில் வைக்கவும்",
                    sizeBadge = "3.44 GB",
                    url = "https://huggingface.co/CompVis/stable-diffusion-v1-4/resolve/onnx/unet/weights.pb",
                    isDirectFile = true
                )
            )
        ),
        OnnxDownloadGroup(
            id = "extra_core_encoders",
            badgeLabelEn = "CORE ENCODERS & TOKENIZER",
            badgeLabelTa = "மற்ற Core மாடல்கள் (Text Encoder & VAE)",
            titleEn = "Core Encoders & Tokenizer ONNX Links",
            titleTa = "மற்ற Core ONNX மாடல்கள் (CLIP & VAE)",
            descriptionEn = "Optional links if you ever need Text Encoder, VAE Decoder/Encoder, or Tokenizer ONNX files.",
            descriptionTa = "Text Encoder, VAE Decoder/Encoder அல்லது Tokenizer மாடல்கள் தேவைப்பட்டால் இங்கே பதிவிறக்கலாம்.",
            isRecommended = false,
            links = listOf(
                OnnxDownloadLinkItem(
                    id = "sd15_full_pipeline_repo",
                    titleEn = "SD 1.5 Full ONNX Pipeline Repo",
                    titleTa = "SD 1.5 முழு ONNX தொகுப்பு பக்கம்",
                    subtitleEn = "text_encoder, vae_decoder, vae_encoder & tokenizer files",
                    subtitleTa = "text_encoder, vae_decoder, vae_encoder & tokenizer ஃபைல்கள்",
                    sizeBadge = "Repo",
                    url = "https://huggingface.co/onnx-community/stable-diffusion-v1-5-ONNX/tree/main",
                    isDirectFile = false
                )
            )
        )
    )
}
