package com.example.ai

import android.content.Context
import org.json.JSONObject
import java.io.File
import java.util.Locale

/**
 * Offline Tamil-to-English creative prompt mapper + OpenAI CLIP BPE Tokenizer (max_length = 77).
 * Uses user-imported tokenizer files if present in filesDir/onnx_models/tokenizer/,
 * otherwise falls back seamlessly to bundled assets/tokenizer/{vocab.json, merges.txt}.
 */
object OfflineTamilPromptTranslator {

    val defaultPhraseMap: List<Pair<String, String>> = listOf(
        // Core safety boilerplate phrases (stripped cleanly into concise English CLIP guidance so 77-token budget isn't wasted)
        "சட்டவிரோதமான தீங்கு விளைவிக்கும் உள்ளடக்கம்" to "",
        "குழந்தைகள் தொடர்பான பாலியல் உள்ளடக்கம்" to "",
        "அனுமதியற்ற உண்மையான நபர் பாலியல் deepfake" to "",
        "போன்றவற்றுக்கு கட்டாய பாதுகாப்பு இருக்க வேண்டும்" to "safe adult artistic quality, natural proportions",
        "மற்றும்" to "and",
        // Creative & Realistic DSLR Tamil Phrases -> English CLIP Descriptors
        "பட்டுப்புடவை" to "traditional silk saree",
        "பட்டுப் புடவை" to "traditional silk saree",
        "பட்டு புடவை" to "traditional silk saree",
        "புடவை" to "indian saree",
        "சுடிதார்" to "salwar kameez outfit",
        "பாவாடை தாவணி" to "traditional half saree",
        "தாவணி" to "traditional half saree",
        "பட்டு வேட்டி சட்டை" to "traditional silk veshti and crisp shirt",
        "வேட்டி சட்டை" to "traditional veshti and collared shirt",
        "வேட்டி" to "traditional south indian veshti",
        "பட்டு குர்தா" to "festive silk kurta",
        "குர்தா" to "tailored kurta",
        "கோட் சூட்" to "tailored formal business suit blazer",
        "ஃபார்மல் சூட்" to "tailored formal business suit",
        "சூட்" to "tailored formal suit",
        "டெனிம் ஜாக்கெட்" to "stitched indigo denim jacket",
        "ஜீன்ஸ்" to "stitched denim jeans",
        "சட்டை" to "tailored collared shirt",
        "டி-ஷர்ட்" to "fitted cotton t-shirt",
        "மாடல் உடை" to "modern fashion outfit",
        "பருத்தி" to "breathable crisp cotton fabric",
        "பட்டு" to "authentic woven silk fabric",
        "லினன்" to "natural organic linen fabric",
        "கரை" to "woven border",
        "பார்டர்" to "intricate woven border",
        "சிக்ஸ்-பேக்" to "toned athletic six-pack abdominal muscles",
        "சிக்ஸ் பேக்" to "toned athletic six-pack abdominal muscles",
        "தசைப்பற்று" to "muscular athletic build",
        "தசை" to "natural athletic muscle definition",
        "உடற்பயிற்சி" to "athletic fitness physique",
        "ஜிம்" to "athletic gym wear",
        "மெலிந்த" to "lean slim fit",
        "வலிமையான" to "strong broad-shouldered",
        "முதியவர்" to "dignified elder with natural aged wrinkles",
        "வயதான" to "mature aged natural wrinkles",
        "இளமை" to "youthful smooth natural skin",
        "சுருக்கம்" to "authentic skin wrinkles",
        "நரை முடி" to "natural salt-and-pepper silver hair",
        "நரைமுடி" to "natural salt-and-pepper silver hair",
        "வெள்ளை முடி" to "natural silver-white hair",
        "சுருட்டை முடி" to "natural voluminous curly hair",
        "அலை அலையான முடி" to "layered wavy hair",
        "நீண்ட கூந்தல்" to "long flowing dark hair",
        "நீளமான முடி" to "long flowing hair",
        "குட்டை முடி" to "clean short textured haircut",
        "பின்னல்" to "neatly braided hairstyle",
        "தாடி" to "well-groomed natural beard",
        "மீசை" to "natural mustache",
        "கைக்கடிகாரம்" to "classic wristwatch",
        "மூக்குக்கண்ணாடி" to "eyeglasses",
        "கண்ணாடி" to "stylish glasses",
        "அணிந்திருக்கும்" to "wearing",
        "அணிந்து" to "wearing",
        "அணிந்த" to "wearing",
        "நிற்கும்" to "standing",
        "அமர்ந்திருக்கும்" to "seated",
        "கடற்கரை" to "scenic tropical beach",
        "சூரிய அஸ்தமனம்" to "warm golden hour sunset",
        "இயற்கை ஒளி" to "soft natural lighting",
        "இயற்கையான" to "natural",
        "சிரிக்கும் முகம்" to "warm natural smile",
        "சிரிக்கும்" to "smiling",
        "புன்னகைக்கும்" to "gently smiling",
        "புன்னகை" to "gentle smile",
        "அழகான" to "beautiful",
        "இளம் பெண்" to "young adult woman",
        "பெண்" to "adult woman",
        "ஆண்" to "adult man",
        "தென்னிந்திய" to "south indian",
        "இந்திய" to "indian",
        "பாரம்பரிய" to "traditional",
        "நவீன" to "modern",
        "கோவில்" to "ancient carved temple architecture",
        "பூங்கா" to "blooming garden",
        "தோட்டம்" to "lush garden",
        "பின்புலத்தில்" to "in the background",
        "பின்புலம்" to "background",
        "மலர்கள்" to "fresh flowers",
        "மல்லிகை" to "jasmine flowers in hair",
        "கூந்தல்" to "flowing hair",
        "கண்கள்" to "expressive eyes",
        "மழை" to "gentle rain droplets",
        "காடு" to "lush green forest",
        "மலை" to "misty mountains",
        "அரண்மனை" to "royal palace interior",
        "இரவு" to "night lighting bokeh",
        "பொக்கே" to "soft bokeh",
        "தெளிவான புகைப்படம்" to "ultra detailed realistic photograph",
        "புகைப்படம்" to "realistic photograph, DSLR photo",
        // Full-body / pose / scene / outfit words (editable later in Settings → Tamil அகராதி)
        "தலை முதல் கால் வரை" to "full body, head to toe",
        "முழுநீளம்" to "full body shot, head to toe",
        "முழுநீள" to "full body shot, head to toe",
        "முழு நீள" to "full body shot, head to toe",
        "முழு உடல்" to "full body shot, head to toe",
        "முழு உருவம்" to "full body shot",
        "அமர்ந்து" to "sitting",
        "அமர்ந்த" to "sitting",
        "நடக்கும்" to "walking",
        "நடந்து" to "walking",
        "நீண்ட முடி" to "long hair",
        "சுருள் முடி" to "curly hair",
        "நீண்ட" to "long",
        "கோயில்" to "ancient temple",
        "மாலை நேரம்" to "evening light",
        "சூரிய ஒளி" to "warm sunlight",
        "நிலா" to "moonlight",
        "தெரு" to "street",
        "நகரம்" to "city",
        "வயல்" to "green paddy field",
        "அறை" to "room interior",
        "ஜன்னல்" to "window light",
        "ஸ்டுடியோ" to "studio lighting",
        "லெஹங்கா" to "embroidered lehenga choli",
        "கவுன்" to "elegant evening gown",
        "மேற்கத்திய உடை" to "western outfit",
        "சேலை" to "indian saree",
        "மணப்பெண்" to "indian bride in bridal attire",
        "மல்லிகை பூ" to "jasmine flowers in hair",
        "நகை" to "gold jewelry",
        "ரோஜா" to "rose",
        "உண்மையான" to "realistic",
        "யதார்த்தமான" to "realistic",
        "தத்ரூபமான" to "photorealistic",
        "கேமரா" to "DSLR camera photo",
        "கருப்பு வெள்ளை" to "black and white photo",
        "முகம்" to "face",
        "உடல்" to "body",
        "கைகள்" to "hands",
        "ஓவியம்" to "artistic digital painting",
        "நகைகள்" to "delicate gold jewelry",
        "தங்க" to "golden",
        "சிவப்பு" to "crimson red",
        "மெரூன்" to "deep maroon",
        "நீலம்" to "royal blue",
        "மஞ்சள்" to "golden yellow",
        "பச்சை" to "emerald green",
        "இளஞ்சிவப்பு" to "rose pink",
        "ஊதா" to "royal purple",
        "வெள்ளை" to "pure white",
        "கருப்பு" to "sleek black"
    )

    fun containsTamil(text: String): Boolean =
        text.any { ch -> ch.code in 0x0B80..0x0BFF }

    /** Longest Tamil phrase first, so "இளஞ்சிவப்பு" wins over "சிவப்பு" and "முழு உடல்" over "உடல்". */
    private fun sortForMatching(list: List<Pair<String, String>>): List<Pair<String, String>> =
        list.filter { it.first.isNotBlank() }.distinctBy { it.first }.sortedByDescending { it.first.length }

    @Volatile
    private var activeMap: List<Pair<String, String>> = sortForMatching(defaultPhraseMap)

    /** null = use the default list. Called whenever Settings → Tamil அகராதி changes. */
    fun setUserDictionary(entries: List<Pair<String, String>>?) {
        activeMap = sortForMatching(entries ?: defaultPhraseMap)
    }

    private val tamilWordRegex = Regex("[\\u0B80-\\u0BFF]+")

    /** Tamil words that are NOT in the dictionary (they would be dropped before reaching the AI). */
    fun unknownTamilWords(rawPrompt: String, dictionary: List<Pair<String, String>>? = null): List<String> {
        var working = rawPrompt
        for ((ta, _) in dictionary?.let(::sortForMatching) ?: activeMap) {
            working = working.replace(ta, " ", ignoreCase = true)
        }
        return tamilWordRegex.findAll(working).map { it.value }.distinct().toList()
    }

    fun normalizeForClip(rawPrompt: String, dictionary: List<Pair<String, String>>? = null): String {
        if (!containsTamil(rawPrompt)) return rawPrompt.trim()

        var working = rawPrompt
        for ((ta, en) in dictionary?.let(::sortForMatching) ?: activeMap) {
            if (working.contains(ta, ignoreCase = true)) {
                working = working.replace(ta, " $en ", ignoreCase = true)
            }
        }
        // Remove any remaining untranslated Tamil script codepoints so CLIP BPE stays clean
        val cleaned = StringBuilder()
        for (ch in working) {
            if (ch.code !in 0x0B80..0x0BFF) {
                cleaned.append(ch)
            } else {
                cleaned.append(' ')
            }
        }
        return cleaned.toString()
            .replace("|", ",")
            .replace(Regex("\\s+"), " ")
            .replace(Regex("(,\\s*)+"), ", ")
            .trim(' ', ',', '.')
            .ifBlank { "realistic photograph, natural lighting, high detail" }
    }
}

class ClipTokenizer(private val context: Context) {

    private var vocab: Map<String, Int>? = null
    private var bpeRanks: Map<Pair<String, String>, Int>? = null
    private var loadedSourceSignature: String = ""
    private val byteEncoder: Map<Int, Char> = buildByteToUnicode()

    companion object {
        const val START_TOKEN_ID = 49406
        const val END_TOKEN_ID = 49407
        const val MAX_LENGTH = 77

        private fun buildByteToUnicode(): Map<Int, Char> {
            val bs = mutableListOf<Int>()
            for (b in '!'.code..'~'.code) bs.add(b)
            for (b in '¡'.code..'¬'.code) bs.add(b)
            for (b in '®'.code..'ÿ'.code) bs.add(b)
            val cs = bs.toMutableList()
            var n = 0
            for (b in 0..255) {
                if (b !in bs) {
                    bs.add(b)
                    cs.add(256 + n)
                    n++
                }
            }
            return bs.zip(cs.map { it.toChar() }).toMap()
        }
    }

    @Synchronized
    private fun ensureLoaded() {
        val customTokDir = OnnxModelManager.resolveTokenizerDir(context)
        val vocabFile = File(customTokDir, "vocab.json")
        val mergesFile = File(customTokDir, "merges.txt")
        val currentSig = "${vocabFile.exists()}:${vocabFile.length()}:${vocabFile.lastModified()}|" +
            "${mergesFile.exists()}:${mergesFile.length()}:${mergesFile.lastModified()}"

        if (vocab != null && bpeRanks != null && loadedSourceSignature == currentSig) return

        val vocabJsonText = runCatching {
            if (vocabFile.exists() && vocabFile.length() > 1024L) vocabFile.readText()
            else context.assets.open("tokenizer/vocab.json").bufferedReader().use { it.readText() }
        }.getOrElse { "{}" }

        val mergesLines = runCatching {
            if (mergesFile.exists() && mergesFile.length() > 1024L) mergesFile.readLines()
            else context.assets.open("tokenizer/merges.txt").bufferedReader().use { it.readLines() }
        }.getOrElse { emptyList() }

        val parsedVocab = HashMap<String, Int>(52000)
        val json = JSONObject(vocabJsonText)
        val keys = json.keys()
        while (keys.hasNext()) {
            val k = keys.next()
            parsedVocab[k] = json.getInt(k)
        }

        val parsedRanks = HashMap<Pair<String, String>, Int>(52000)
        var rank = 0
        for (line in mergesLines) {
            if (line.isBlank() || line.startsWith("#")) continue
            val parts = line.trim().split(Regex("\\s+"))
            if (parts.size == 2) {
                parsedRanks[parts[0] to parts[1]] = rank++
            }
        }

        vocab = parsedVocab
        bpeRanks = parsedRanks
        loadedSourceSignature = currentSig
    }

    fun encode(prompt: String): IntArray {
        ensureLoaded()
        val vocabMap = vocab ?: emptyMap()
        val normalized = OfflineTamilPromptTranslator.normalizeForClip(prompt)
            .lowercase(Locale.US)
            .replace(Regex("\\s+"), " ")
            .trim()

        val tokens = mutableListOf(START_TOKEN_ID)

        if (normalized.isNotEmpty() && vocabMap.isNotEmpty()) {
            val wordPattern = Regex("""'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""")
            for (match in wordPattern.findAll(normalized)) {
                val piece = match.value
                val encodedPiece = buildString {
                    for (byte in piece.toByteArray(Charsets.UTF_8)) {
                        append(byteEncoder[byte.toInt() and 0xFF] ?: ' ')
                    }
                }
                for (bpeToken in bpe(encodedPiece)) {
                    val id = vocabMap[bpeToken]
                    if (id != null) {
                        tokens.add(id)
                        if (tokens.size >= MAX_LENGTH - 1) break
                    }
                }
                if (tokens.size >= MAX_LENGTH - 1) break
            }
        }

        tokens.add(END_TOKEN_ID)
        while (tokens.size < MAX_LENGTH) {
            tokens.add(END_TOKEN_ID)
        }
        return tokens.take(MAX_LENGTH).toIntArray()
    }

    private fun bpe(token: String): List<String> {
        val ranks = bpeRanks ?: return listOf("$token</w>")
        if (token.isEmpty()) return emptyList()
        if (token.length == 1) return listOf("$token</w>")

        var word = token.dropLast(1).map { it.toString() } + listOf("${token.last()}</w>")

        while (word.size > 1) {
            var bestPair: Pair<String, String>? = null
            var bestRank = Int.MAX_VALUE
            for (i in 0 until word.size - 1) {
                val pair = word[i] to word[i + 1]
                val r = ranks[pair]
                if (r != null && r < bestRank) {
                    bestRank = r
                    bestPair = pair
                }
            }
            if (bestPair == null) break

            val (first, second) = bestPair
            val nextWord = mutableListOf<String>()
            var i = 0
            while (i < word.size) {
                if (i < word.size - 1 && word[i] == first && word[i + 1] == second) {
                    nextWord.add(first + second)
                    i += 2
                } else {
                    nextWord.add(word[i])
                    i += 1
                }
            }
            word = nextWord
        }
        return word
    }
}
