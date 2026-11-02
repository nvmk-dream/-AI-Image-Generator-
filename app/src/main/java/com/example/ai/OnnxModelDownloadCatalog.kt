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
            id = "lcm_fast_mobile_fp32",
            badgeLabelEn = "RECOMMENDED • FAST MOBILE (4–6 STEPS)",
            badgeLabelTa = "பரிந்துரைக்கப்படுகிறது • மொபைலுக்கு ஏற்றது (4–6 படிகள்)",
            titleEn = "Option 1: Fast LCM FP32 UNet (Mobile Compatible)",
            titleTa = "விருப்பம் 1: மொபைலில் இயங்கும் Fast LCM FP32 UNet",
            descriptionEn = "Runs in 4–6 steps without ConvInteger errors on Android ARM64. Download model.onnx and model.onnx_data into 'unet' folder without renaming.",
            descriptionTa = "மொபைல் பிராசஸரில் (ARM64) 4–6 படிகளில் வேகமாக இயங்கக்கூடிய Fast LCM மாடல். இதில் ConvInteger எரர் வராது. 'unet' ஃபோல்டரில் சேமிக்கவும்.",
            isRecommended = true,
            links = listOf(
                OnnxDownloadLinkItem(
                    id = "lcm_hex_unet_folder",
                    titleEn = "Fast LCM UNet Folder (TheyCallMeHex)",
                    titleTa = "Fast LCM UNet பக்கம் (TheyCallMeHex)",
                    subtitleEn = "Compatible FP32 Fast LCM UNet directory",
                    subtitleTa = "மொபைலுக்கு ஏற்ற Fast LCM UNet ஃபைல்கள்",
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
                    titleEn = "Full LCM Pipeline Repo (All Core Models)",
                    titleTa = "முழு LCM தொகுப்பு பக்கம் (அனைத்து மாடல்கள்)",
                    subtitleEn = "Includes text_encoder, unet, vae_decoder, vae_encoder",
                    subtitleTa = "text_encoder, unet, vae_decoder, vae_encoder அனைத்தும் உள்ளது",
                    sizeBadge = "Full Repo",
                    url = "https://huggingface.co/TheyCallMeHex/LCM-Dreamshaper-V7-ONNX/tree/main",
                    isDirectFile = false
                )
            )
        ),
        OnnxDownloadGroup(
            id = "lcm_int8_pc_warning",
            badgeLabelEn = "⚠️ PC ONLY • NOT FOR MOBILE ARM64",
            badgeLabelTa = "⚠️ கணினிகளுக்கு மட்டும் • மொபைலில் இயங்காது",
            titleEn = "Option 3: Fast LCM INT8 (~860 MB - x86 PC Only)",
            titleTa = "விருப்பம் 3: Fast LCM INT8 (~860 MB - PC மட்டும்)",
            descriptionEn = "Warning: Contains ConvInteger(10) unsupported on mobile ARM64 processors. Runs on x86 emulators/desktops only.",
            descriptionTa = "எச்சரிக்கை: இதில் ConvInteger(10) இருப்பதால் மொபைல் போன்களில் இயங்காது. கணினி அல்லது x86 எமுலேட்டர்களில் மட்டுமே இயங்கும்.",
            isRecommended = false,
            links = listOf(
                OnnxDownloadLinkItem(
                    id = "lcm_unet_folder",
                    titleEn = "INT8 UNet Folder (PC Only)",
                    titleTa = "INT8 UNet பக்கம் (PC மட்டும்)",
                    subtitleEn = "x86 PC only — causes ConvInteger error on mobile phones",
                    subtitleTa = "கணினிக்கு மட்டும் — மொபைல் போனில் ConvInteger எரர் வரும்",
                    sizeBadge = "Folder",
                    url = "https://huggingface.co/latentdivergence/lcm-dreamshaper-int8-onnx/tree/main/unet",
                    isDirectFile = false
                ),
                OnnxDownloadLinkItem(
                    id = "lcm_unet_graph",
                    titleEn = "1. model.onnx (Graph)",
                    titleTa = "1. model.onnx (கட்டமைப்பு)",
                    subtitleEn = "INT8 graph shell",
                    subtitleTa = "INT8 கட்டமைப்பு ஃபைல்",
                    sizeBadge = "~2 MB",
                    url = "https://huggingface.co/latentdivergence/lcm-dreamshaper-int8-onnx/resolve/main/unet/model.onnx",
                    isDirectFile = true
                ),
                OnnxDownloadLinkItem(
                    id = "lcm_unet_data",
                    titleEn = "2. model.onnx.data (INT8 Weights)",
                    titleTa = "2. model.onnx.data (INT8 எடை ஃபைல்)",
                    subtitleEn = "ConvInteger weights — PC only",
                    subtitleTa = "ConvInteger ஃபைல் — PC-க்கு மட்டும்",
                    sizeBadge = "~860 MB",
                    url = "https://huggingface.co/latentdivergence/lcm-dreamshaper-int8-onnx/resolve/main/unet/model.onnx.data",
                    isDirectFile = true
                )
            )
        ),
        OnnxDownloadGroup(
            id = "sd15_fp32_weights_pb",
            badgeLabelEn = "STANDARD • FULL FP32",
            badgeLabelTa = "சாதாரண முழு அளவு • FP32 (3.44 GB)",
            titleEn = "Option 2: Standard SD UNet (model.onnx + weights.pb)",
            titleTa = "விருப்பம் 2: முழு FP32 UNet (model.onnx + weights.pb)",
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
            id = "extra_core_and_faceswap",
            badgeLabelEn = "CORE & FACE SWAP ADD-ONS",
            badgeLabelTa = "மற்ற மாடல்கள் & Face Swap",
            titleEn = "Core Encoders & Face Swap ONNX Links",
            titleTa = "மற்ற Core & Face Swap ONNX மாடல்கள்",
            descriptionEn = "Optional links if you ever need Text Encoder, VAE Decoder/Encoder, or Face Swap ONNX files.",
            descriptionTa = "Text Encoder, VAE அல்லது Face Swap மாடல்கள் தேவைப்பட்டால் இங்கே பதிவிறக்கலாம்.",
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
                ),
                OnnxDownloadLinkItem(
                    id = "faceswap_inswapper_repo",
                    titleEn = "InSwapper 128 ONNX (Face Swap)",
                    titleTa = "InSwapper 128 ONNX (முகம் மாற்றும் மாடல்)",
                    subtitleEn = "inswapper_128.onnx for offline Face Swap slot #7",
                    subtitleTa = "Face Swap மாடல் #7-க்கான inswapper_128.onnx",
                    sizeBadge = "~529 MB",
                    url = "https://huggingface.co/ezioruan/inswapper_128.onnx/tree/main",
                    isDirectFile = false
                ),
                OnnxDownloadLinkItem(
                    id = "faceswap_buffalo_l_repo",
                    titleEn = "InsightFace Buffalo_L (det_10g & w600k_r50)",
                    titleTa = "Face Detector & Identity (det_10g & w600k_r50)",
                    subtitleEn = "SCRFD det_10g.onnx + ArcFace w600k_r50.onnx models",
                    subtitleTa = "Face Swap மாடல் #5 மற்றும் #6 ஃபைல்கள்",
                    sizeBadge = "Repo",
                    url = "https://huggingface.co/public-data/insightface/tree/main/models/buffalo_l",
                    isDirectFile = false
                )
            )
        )
    )
}
