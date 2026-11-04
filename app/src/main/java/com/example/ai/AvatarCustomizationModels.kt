package com.example.ai

import androidx.annotation.DrawableRes
import com.example.R
import java.util.Locale
import kotlin.math.abs

enum class AvatarCustomTab(
    val id: String,
    val titleEn: String,
    val titleTa: String,
    val shortTitleEn: String,
    val shortTitleTa: String
) {
    AGE(
        id = "age",
        titleEn = "1. AI Age Progression",
        titleTa = "1. வயது மாற்றம் (Age)",
        shortTitleEn = "Age",
        shortTitleTa = "வயது"
    ),
    BODY_ABS(
        id = "body_abs",
        titleEn = "2. AI Body / Abs",
        titleTa = "2. உடலமைப்பு (Body/Abs)",
        shortTitleEn = "Body/Abs",
        shortTitleTa = "உடல்/Abs"
    ),
    HAIRSTYLE(
        id = "hairstyle",
        titleEn = "3. AI Hairstyle",
        titleTa = "3. சிகை அலங்காரம் (Hair)",
        shortTitleEn = "Hairstyle",
        shortTitleTa = "முடி"
    ),
    OUTFIT(
        id = "outfit",
        titleEn = "4. AI Outfit / Clothes",
        titleTa = "4. ஆடை மாற்றம் (Outfit)",
        shortTitleEn = "Outfit",
        shortTitleTa = "ஆடை"
    )
}

enum class HairShadeOption(
    val id: String,
    val labelEn: String,
    val labelTa: String,
    val promptPhrase: String
) {
    NATURAL_BLACK("natural_black", "Natural Jet Black", "இயற்கை கருப்பு", "natural deep jet-black human hair"),
    ESPRESSO_BROWN("espresso_brown", "Dark Espresso", "அடர் பழுப்பு", "natural dark espresso-brown human hair"),
    WARM_CHESTNUT("warm_chestnut", "Warm Chestnut", "செம்பழுப்பு", "subtle warm chestnut-brown sunlit human hair"),
    SALT_PEPPER("salt_pepper", "Salt & Pepper", "நரை கலந்த முடி", "dignified natural salt-and-pepper silver-streaked hair")
}

enum class OutfitFabricOption(
    val id: String,
    val labelEn: String,
    val labelTa: String,
    val promptPhrase: String
) {
    PURE_SILK("pure_silk", "Pure Woven Silk", "பட்டு (Silk)", "authentic handwoven pure silk fabric with natural sheen and realistic folds"),
    CRISP_COTTON("crisp_cotton", "Crisp Cotton", "பருத்தி (Cotton)", "breathable natural woven cotton fabric with realistic creases"),
    TAILORED_WOOL("tailored_wool", "Tailored Suit Wool", "சூட் துணி (Wool)", "fine tailored suiting wool cloth with structured lapel drape"),
    INDIGO_DENIM("indigo_denim", "Natural Denim", "டெனிம் (Denim)", "authentic stitched cotton denim fabric with realistic seams"),
    RAW_LINEN("raw_linen", "Organic Linen", "லினன் (Linen)", "textured natural organic linen fabric with candid daylight folds")
}

data class RealisticAvatarPreset(
    val id: String,
    val tab: AvatarCustomTab,
    val titleEn: String,
    val titleTa: String,
    val badgeEn: String,
    val badgeTa: String,
    @param:DrawableRes val photoResId: Int,
    val promptDescriptor: String,
    val defaultTargetAge: Int? = null,
    val defaultWrinkleIntensity: Float? = null,
    val defaultSilverHair: Float? = null,
    val defaultAbsDefinition: Float? = null,
    val defaultMuscleMass: Float? = null,
    val defaultWaistTrim: Float? = null,
    val defaultHairVolume: Float? = null,
    val defaultHairLength: Float? = null,
    val defaultClothDraping: Float? = null,
    val defaultStrength: Float = 0.52f
)

data class MaskBrushStroke(
    val normalizedPoints: List<Pair<Float, Float>>,
    val normalizedRadius: Float,
    val isErase: Boolean
)

/**
 * Holds all 16 user-controlled customization sliders + preset/chip selections.
 * Strictly governs realistic DSLR photography editing only.
 */
data class AvatarCustomizationParams(
    val activeTab: AvatarCustomTab = AvatarCustomTab.AGE,
    val selectedPresetId: String = "age_senior_72",

    // --- Tab 1: AI Age Progression Sliders (4) ---
    val targetAge: Int = 65,                        // Slider 1: 16..80 years
    val skinWrinkleIntensity: Float = 0.68f,        // Slider 2: 0f..1f
    val silverHairRatio: Float = 0.60f,             // Slider 3: 0f..1f
    val facialBoneConsistency: Float = 0.85f,       // Slider 4: 0.4f..0.98f

    // --- Tab 2: AI Body / Abs Sliders (3) ---
    val absCoreDefinition: Float = 0.72f,           // Slider 5: 0f..1f
    val muscleShoulderMass: Float = 0.58f,          // Slider 6: 0f..1f
    val waistBodyTrim: Float = 0.50f,               // Slider 7: 0f..1f

    // --- Tab 3: AI Hairstyle Sliders (3) + Hair Shade ---
    val hairVolumeDensity: Float = 0.65f,           // Slider 8: 0f..1f
    val hairLengthScale: Float = 0.60f,             // Slider 9: 0f..1f
    val hairNaturalShine: Float = 0.55f,            // Slider 10: 0f..1f
    val hairShade: HairShadeOption = HairShadeOption.NATURAL_BLACK,

    // --- Tab 4: AI Outfit / Clothes Sliders (2) + Fabric ---
    val clothDrapingRealism: Float = 0.75f,         // Slider 11: 0f..1f
    val garmentTailoredFit: Float = 0.60f,          // Slider 12: 0f..1f
    val outfitFabric: OutfitFabricOption = OutfitFabricOption.PURE_SILK,

    // --- Shared Mask, Identity & Realism Studio Sliders (4 -> Total = 16 Sliders!) ---
    val transformationStrength: Float = 0.50f,      // Slider 13: 0.25f..0.82f
    val faceIdentityLock: Float = 0.90f,            // Slider 14: 0.50f..1.00f
    val maskFeatherPx: Float = 16f,                 // Slider 15: 4f..32f
    val maskExpandPx: Float = 4f,                   // Slider 16: -8f..24f

    // Optional user realistic detail text + seed
    val customRealisticNote: String = "",
    val lockSeed: Boolean = true,
    val seed: Long = 428_519L
)

object AvatarCustomizationCatalog {

    val presets: List<RealisticAvatarPreset> = listOf(
        // ==================== TAB 1: AI AGE PROGRESSION ====================
        RealisticAvatarPreset(
            id = "age_teen_18",
            tab = AvatarCustomTab.AGE,
            titleEn = "18 Yrs • Youthful",
            titleTa = "18 வயது • இளமை",
            badgeEn = "Age 18",
            badgeTa = "18 வயது",
            photoResId = R.drawable.img_preset_age_teen,
            promptDescriptor = "18 year old youthful adult face, smooth healthy natural skin pores, vibrant clear eyes, thick natural dark hair, candid DSLR portrait photograph",
            defaultTargetAge = 18,
            defaultWrinkleIntensity = 0.08f,
            defaultSilverHair = 0.0f,
            defaultStrength = 0.38f
        ),
        RealisticAvatarPreset(
            id = "age_young_25",
            tab = AvatarCustomTab.AGE,
            titleEn = "25 Yrs • Prime",
            titleTa = "25 வயது • பொலிவு",
            badgeEn = "Age 25",
            badgeTa = "25 வயது",
            photoResId = R.drawable.img_preset_age_youth,
            promptDescriptor = "25 year old young adult face, natural skin texture and pores, radiant healthy complexion, realistic DSLR portrait photography",
            defaultTargetAge = 25,
            defaultWrinkleIntensity = 0.14f,
            defaultSilverHair = 0.0f,
            defaultStrength = 0.36f
        ),
        RealisticAvatarPreset(
            id = "age_middle_45",
            tab = AvatarCustomTab.AGE,
            titleEn = "45 Yrs • Mature",
            titleTa = "45 வயது • நடுத்தர",
            badgeEn = "Age 45",
            badgeTa = "45 வயது",
            photoResId = R.drawable.img_preset_age_middle,
            promptDescriptor = "45 year old mature adult face, subtle natural expression lines around eyes, slight salt-and-pepper temple hair, authentic skin pores, candid 85mm DSLR photograph",
            defaultTargetAge = 45,
            defaultWrinkleIntensity = 0.42f,
            defaultSilverHair = 0.28f,
            defaultStrength = 0.44f
        ),
        RealisticAvatarPreset(
            id = "age_senior_72",
            tab = AvatarCustomTab.AGE,
            titleEn = "72 Yrs • Elder",
            titleTa = "72 வயது • முதியவர்",
            badgeEn = "Age 72",
            badgeTa = "72 வயது",
            photoResId = R.drawable.img_preset_age_senior,
            promptDescriptor = "72 year old dignified elder face, realistic natural forehead wrinkles, crow's feet, nasolabial folds, natural silver-grey hair, genuine aged human skin texture, 85mm DSLR portrait photograph",
            defaultTargetAge = 72,
            defaultWrinkleIntensity = 0.78f,
            defaultSilverHair = 0.80f,
            defaultStrength = 0.50f
        ),

        // ==================== TAB 2: AI BODY / ABS ====================
        RealisticAvatarPreset(
            id = "body_six_pack_abs",
            tab = AvatarCustomTab.BODY_ABS,
            titleEn = "Six-Pack Abs",
            titleTa = "சிக்ஸ்-பேக் Abs",
            badgeEn = "Athletic",
            badgeTa = "சிக்ஸ்-பேக்",
            photoResId = R.drawable.img_preset_body_abs,
            promptDescriptor = "toned athletic fit human torso with natural realistic six-pack abdominal muscle definition, natural skin pores and realistic daylight shadows, authentic DSLR fitness photograph",
            defaultAbsDefinition = 0.82f,
            defaultMuscleMass = 0.65f,
            defaultWaistTrim = 0.68f,
            defaultStrength = 0.56f
        ),
        RealisticAvatarPreset(
            id = "body_toned_fit",
            tab = AvatarCustomTab.BODY_ABS,
            titleEn = "Toned Fitness",
            titleTa = "ஃபிட் உடலமைப்பு",
            badgeEn = "Toned",
            badgeTa = "ஃபிட்",
            photoResId = R.drawable.img_preset_body_toned,
            promptDescriptor = "healthy toned athletic fit body posture in realistic activewear, proportionate natural shoulders and trim waist, candid outdoor DSLR photograph",
            defaultAbsDefinition = 0.55f,
            defaultMuscleMass = 0.48f,
            defaultWaistTrim = 0.60f,
            defaultStrength = 0.52f
        ),
        RealisticAvatarPreset(
            id = "body_muscular_build",
            tab = AvatarCustomTab.BODY_ABS,
            titleEn = "Muscular Frame",
            titleTa = "தசைப்பற்று உடல்",
            badgeEn = "Strong",
            badgeTa = "வலிமை",
            photoResId = R.drawable.img_preset_body_muscular,
            promptDescriptor = "strong broad-shouldered muscular athletic build with realistic human muscle anatomy and natural skin texture in fitted gym attire, authentic DSLR photograph",
            defaultAbsDefinition = 0.74f,
            defaultMuscleMass = 0.85f,
            defaultWaistTrim = 0.55f,
            defaultStrength = 0.56f
        ),
        RealisticAvatarPreset(
            id = "body_lean_slim",
            tab = AvatarCustomTab.BODY_ABS,
            titleEn = "Lean & Slim",
            titleTa = "மெலிந்த ஃபிட்",
            badgeEn = "Lean",
            badgeTa = "மெலிவு",
            photoResId = R.drawable.img_preset_body_slim,
            promptDescriptor = "lean slim athletic runner physique with natural tailored posture and trim waistline, authentic human photography in natural daylight",
            defaultAbsDefinition = 0.42f,
            defaultMuscleMass = 0.35f,
            defaultWaistTrim = 0.80f,
            defaultStrength = 0.50f
        ),

        // ==================== TAB 3: AI HAIRSTYLE ====================
        RealisticAvatarPreset(
            id = "hair_layered_waves",
            tab = AvatarCustomTab.HAIRSTYLE,
            titleEn = "Layered Waves",
            titleTa = "அலை அலையான முடி",
            badgeEn = "Long Wave",
            badgeTa = "நீள முடி",
            photoResId = R.drawable.img_preset_hair_waves,
            promptDescriptor = "voluminous long layered wavy hair with individual realistic hair strands and natural hairline blended seamlessly, 85mm DSLR portrait photograph",
            defaultHairVolume = 0.78f,
            defaultHairLength = 0.82f,
            defaultStrength = 0.56f
        ),
        RealisticAvatarPreset(
            id = "hair_jasmine_braid",
            tab = AvatarCustomTab.HAIRSTYLE,
            titleEn = "Jasmine Braid",
            titleTa = "மல்லிகை பின்னல்",
            badgeEn = "Traditional",
            badgeTa = "பாரம்பரியம்",
            photoResId = R.drawable.img_preset_hair_braid,
            promptDescriptor = "sleek traditional neatly parted dark braided hairstyle adorned with fresh white jasmine flowers, individual realistic hair strands, candid DSLR photograph",
            defaultHairVolume = 0.62f,
            defaultHairLength = 0.85f,
            defaultStrength = 0.55f
        ),
        RealisticAvatarPreset(
            id = "hair_modern_fade",
            tab = AvatarCustomTab.HAIRSTYLE,
            titleEn = "Textured Fade",
            titleTa = "மாடர்ன் ஃபேட் கட்",
            badgeEn = "Short Cut",
            badgeTa = "ஷார்ட் கட்",
            photoResId = R.drawable.img_preset_hair_fade,
            promptDescriptor = "clean modern barber textured crop haircut with natural tapered side fade and realistic hairline strands, authentic DSLR portrait photograph",
            defaultHairVolume = 0.52f,
            defaultHairLength = 0.22f,
            defaultStrength = 0.55f
        ),
        RealisticAvatarPreset(
            id = "hair_natural_curls",
            tab = AvatarCustomTab.HAIRSTYLE,
            titleEn = "Natural Curls",
            titleTa = "சுருட்டை முடி",
            badgeEn = "Curly",
            badgeTa = "சுருட்டை",
            photoResId = R.drawable.img_preset_hair_curls,
            promptDescriptor = "defined voluminous natural curly ringlets with realistic individual hair curls catching soft daylight, candid 85mm DSLR photograph",
            defaultHairVolume = 0.86f,
            defaultHairLength = 0.68f,
            defaultStrength = 0.56f
        ),

        // ==================== TAB 4: AI OUTFIT / CLOTHES ====================
        RealisticAvatarPreset(
            id = "outfit_silk_saree",
            tab = AvatarCustomTab.OUTFIT,
            titleEn = "Royal Silk Saree",
            titleTa = "பட்டுப் புடவை",
            badgeEn = "Traditional",
            badgeTa = "பட்டு",
            photoResId = R.drawable.img_preset_outfit_saree,
            promptDescriptor = "wearing an authentic rich kanjeevaram silk saree with intricate gold zari border and tailored blouse, realistic cloth folds following body pose, natural DSLR photograph",
            defaultClothDraping = 0.82f,
            defaultStrength = 0.58f
        ),
        RealisticAvatarPreset(
            id = "outfit_formal_suit",
            tab = AvatarCustomTab.OUTFIT,
            titleEn = "Business Suit",
            titleTa = "ஃபார்மல் சூட்",
            badgeEn = "Formal",
            badgeTa = "சூட்",
            photoResId = R.drawable.img_preset_outfit_suit,
            promptDescriptor = "wearing a sharp tailored charcoal navy formal business suit blazer over a crisp white collared shirt, natural lapel and sleeve cloth creases, candid DSLR photograph",
            defaultClothDraping = 0.78f,
            defaultStrength = 0.58f
        ),
        RealisticAvatarPreset(
            id = "outfit_casual_denim",
            tab = AvatarCustomTab.OUTFIT,
            titleEn = "Denim Jacket",
            titleTa = "டெனிம் ஜாக்கெட்",
            badgeEn = "Casual",
            badgeTa = "கேஷுவல்",
            photoResId = R.drawable.img_preset_outfit_casual,
            promptDescriptor = "wearing a classic indigo stitched denim jacket layered over a clean white crew-neck cotton t-shirt, realistic fabric seams and folds, outdoor DSLR photograph",
            defaultClothDraping = 0.74f,
            defaultStrength = 0.56f
        ),
        RealisticAvatarPreset(
            id = "outfit_festive_kurta",
            tab = AvatarCustomTab.OUTFIT,
            titleEn = "Festive Silk Kurta",
            titleTa = "பட்டு குர்தா",
            badgeEn = "Festive",
            badgeTa = "குர்தா",
            photoResId = R.drawable.img_preset_outfit_kurta,
            promptDescriptor = "wearing an elegant cream-gold festive raw silk kurta with subtle woven embroidery collar and natural cloth drape, candid daylight DSLR photograph",
            defaultClothDraping = 0.78f,
            defaultStrength = 0.57f
        )
    )

    fun presetsForTab(tab: AvatarCustomTab): List<RealisticAvatarPreset> =
        presets.filter { it.tab == tab }

    fun findPreset(id: String): RealisticAvatarPreset? =
        presets.firstOrNull { it.id == id }

    private val FORBIDDEN_NON_REALISTIC_WORDS = listOf(
        "digital painting", "oil painting", "oil paint", "painting", "watercolor", "gouache", "acrylic",
        "3d render", "3d", "cgi", "render", "blender", "unreal engine", "octane",
        "anime", "manga", "cartoon", "animation", "animated", "pixar", "disney",
        "illustration", "sketch", "drawing", "lineart", "comic", "cel shaded", "vector art", "doll"
    )

    private val FORBIDDEN_TAMIL_NON_REALISTIC_WORDS = listOf(
        "ஓவியம்", "கார்ட்டூன்", "அனிமேஷன்", "சித்திரம்", "பொம்மை"
    )

    const val STRICT_ANTI_NON_PHOTO_NEGATIVE_PROMPT: String =
        "painting, oil painting, watercolor, illustration, sketch, drawing, 3d render, CGI, " +
            "cartoon, anime, manga, animation, Pixar, plastic skin, airbrushed doll, wax figure, " +
            "deformed face, altered identity, asymmetrical eyes, deformed hands, extra fingers, blurry, low quality"

    /**
     * Translates Tamil words to English DSLR descriptors if present, and strips only
     * painting/3D/animation terms so Avatar Customization produces 100% realistic camera photography.
     */
    fun sanitizeCustomRealisticNote(rawNote: String): Pair<String, Boolean> {
        var working = rawNote.trim()
        if (working.isEmpty()) return "" to false

        var strippedAny = false
        for (taBad in FORBIDDEN_TAMIL_NON_REALISTIC_WORDS) {
            if (working.contains(taBad)) {
                strippedAny = true
                working = working.replace(taBad, " ")
            }
        }

        if (OfflineTamilPromptTranslator.containsTamil(working)) {
            working = OfflineTamilPromptTranslator.normalizeForClip(working)
        }

        for (bad in FORBIDDEN_NON_REALISTIC_WORDS) {
            val regex = Regex("(?<![a-z0-9])${Regex.escape(bad)}(?![a-z0-9])", RegexOption.IGNORE_CASE)
            if (regex.containsMatchIn(working)) {
                strippedAny = true
                working = working.replace(regex, " ").trim()
            }
        }

        working = working
            .replace(Regex("\\bartistic\\b", RegexOption.IGNORE_CASE), " ")
            .replace(Regex("\\s{2,}"), " ")
            .replace(Regex("(,\\s*)+"), ", ")
            .trim(' ', ',', '.')
        return working to strippedAny
    }

    /**
     * Builds the camera-realistic positive prompt (front-loaded within CLIP's 77-token window)
     * and calibrated masked-region denoising strength from all 16 user sliders.
     */
    fun buildRealisticPromptAndStrength(params: AvatarCustomizationParams): Pair<String, Float> {
        val preset = findPreset(params.selectedPresetId)
            ?.takeIf { it.tab == params.activeTab }
            ?: presetsForTab(params.activeTab).first()

        val (cleanNote, _) = sanitizeCustomRealisticNote(params.customRealisticNote)
        val noteClause = if (cleanNote.isNotBlank()) ", $cleanNote" else ""

        return when (params.activeTab) {
            AvatarCustomTab.AGE -> {
                val age = params.targetAge.coerceIn(16, 80)
                val wrinkleDesc = when {
                    age <= 24 || params.skinWrinkleIntensity < 0.22f ->
                        "smooth youthful natural skin pores, clear eyes"
                    age <= 42 || params.skinWrinkleIntensity < 0.48f ->
                        "natural adult skin texture, subtle expression lines"
                    age <= 62 || params.skinWrinkleIntensity < 0.72f ->
                        "mature skin creases, natural laugh lines and crow's feet"
                    else ->
                        "deep authentic aged skin wrinkles, forehead lines, natural crow's feet"
                }
                val hairAgeDesc = when {
                    params.silverHairRatio >= 0.70f -> "natural silver-white hair strands"
                    params.silverHairRatio >= 0.35f -> "salt-and-pepper greying temple hair"
                    else -> "natural dark human hair"
                }
                val ageDeltaFactor = (abs(age - 30).toFloat() / 50f).coerceIn(0.10f, 1.0f)
                val computedStrength = (
                    0.34f +
                        params.transformationStrength * 0.42f +
                        ageDeltaFactor * 0.12f -
                        (params.facialBoneConsistency - 0.75f) * 0.12f
                    ).coerceIn(0.36f, 0.72f)

                val prompt = "RAW 85mm DSLR photograph of same person at age $age years old$noteClause, " +
                    "$wrinkleDesc, $hairAgeDesc, ${preset.promptDescriptor}, photorealistic photography"
                prompt to computedStrength
            }

            AvatarCustomTab.BODY_ABS -> {
                val absDesc = when {
                    params.absCoreDefinition >= 0.72f -> "well-defined natural six-pack abdominal muscles, sculpted core"
                    params.absCoreDefinition >= 0.42f -> "toned athletic flat midsection, fit core"
                    else -> "natural healthy proportionate torso"
                }
                val massDesc = when {
                    params.muscleShoulderMass >= 0.72f -> "broad muscular shoulders and athletic chest"
                    params.muscleShoulderMass >= 0.42f -> "toned athletic shoulders and upper body"
                    else -> "lean streamlined athletic frame"
                }
                val waistDesc = if (params.waistBodyTrim >= 0.60f) ", trim athletic waistline" else ""
                val computedStrength = (
                    0.38f + params.transformationStrength * 0.45f + params.absCoreDefinition * 0.08f
                    ).coerceIn(0.46f, 0.84f)

                val prompt = "RAW candid DSLR photograph, $absDesc, $massDesc$waistDesc$noteClause, " +
                    "${preset.promptDescriptor}, natural skin pores and daylight, photorealistic photography"
                prompt to computedStrength
            }

            AvatarCustomTab.HAIRSTYLE -> {
                val volDesc = when {
                    params.hairVolumeDensity >= 0.72f -> "thick voluminous high-density hair"
                    params.hairVolumeDensity >= 0.40f -> "natural medium-volume hair"
                    else -> "sleek close-fitted neat hair"
                }
                val lenDesc = when {
                    params.hairLengthScale >= 0.70f -> "flowing long length"
                    params.hairLengthScale >= 0.38f -> "shoulder-to-medium length"
                    else -> "clean short cropped haircut"
                }
                val shineDesc = if (params.hairNaturalShine >= 0.60f) {
                    "natural daylight hair sheen"
                } else {
                    "natural matte hair texture"
                }
                val computedStrength = (0.40f + params.transformationStrength * 0.46f).coerceIn(0.48f, 0.85f)

                val prompt = "RAW 85mm DSLR photograph, ${preset.promptDescriptor}$noteClause, " +
                    "$volDesc, $lenDesc, ${params.hairShade.promptPhrase}, $shineDesc, " +
                    "realistic hair strands and hairline, photorealistic photography"
                prompt to computedStrength
            }

            AvatarCustomTab.OUTFIT -> {
                val foldDesc = if (params.clothDrapingRealism >= 0.65f) {
                    "natural gravity cloth folds, realistic seams and ambient fabric shadows"
                } else {
                    "clean smooth tailored garment folds"
                }
                val fitDesc = when {
                    params.garmentTailoredFit >= 0.70f -> "bespoke tailored slim fit"
                    params.garmentTailoredFit >= 0.38f -> "classic comfortable natural fit"
                    else -> "relaxed breathable garment fit"
                }
                val computedStrength = (0.42f + params.transformationStrength * 0.46f).coerceIn(0.50f, 0.86f)

                val prompt = "RAW candid DSLR photograph, ${preset.promptDescriptor}$noteClause, " +
                    "made of ${params.outfitFabric.promptPhrase}, $fitDesc, $foldDesc, " +
                    "authentic woven textile texture, photorealistic photography"
                prompt to computedStrength
            }
        }
    }

    fun formatPercent(value: Float): String =
        String.format(Locale.US, "%d%%", (value.coerceIn(0f, 1f) * 100f).toInt())
}
