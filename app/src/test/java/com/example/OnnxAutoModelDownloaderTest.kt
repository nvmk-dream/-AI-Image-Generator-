package com.example

import android.content.Context
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.test.core.app.ApplicationProvider
import com.example.ai.DedicatedModelSlot
import com.example.ai.OnnxAutoModelDownloader
import com.example.ai.OnnxModelManager
import com.example.ai.RealDeviceAndModelMetrics
import com.example.ui.components.FirstLaunchModelDownloadDialog
import com.example.ui.theme.MyApplicationTheme
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [36])
class OnnxAutoModelDownloaderTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    @Test
    fun `first open immediately creates IG_Models and all core slot subfolders`() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val manager = OnnxModelManager(context)
        manager.deleteAllModels()

        val root = manager.ensurePhoneModelFoldersCreated()
        assertTrue(root.exists() && root.isDirectory)
        assertEquals(OnnxModelManager.PHONE_MODELS_DIR_NAME, root.name)

        val expectedSubfolders = listOf(
            "unet_fast_lcm",
            "unet_standard_sd",
            "text_encoder",
            "vae_decoder",
            "vae_encoder",
            "tokenizer"
        )
        for (sub in expectedSubfolders) {
            val dir = File(root, sub)
            assertTrue("Expected subfolder $sub to exist immediately on launch", dir.exists() && dir.isDirectory)
        }
    }

    @Test
    fun `app update preserves previously uploaded Build 16 models in legacy onnx_models folder`() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val manager = OnnxModelManager(context)
        manager.deleteAllModels()

        // Simulate Build 16 having stored Slot 3 (Text Encoder) and Slot 6 (Tokenizer) in legacy filesDir/onnx_models
        val legacyRoot = manager.legacyInternalRootDir
        val legacyTextDir = File(legacyRoot, "text_encoder").apply { mkdirs() }
        val validOnnxBytes = buildValidOnnxHeaderPayload(size = 8192)
        File(legacyTextDir, "model.onnx").writeBytes(validOnnxBytes)
        File(legacyTextDir, "source_name.meta").writeText("model (3).onnx")

        val legacyTokDir = File(legacyRoot, "tokenizer").apply { mkdirs() }
        File(legacyTokDir, "vocab.json").writeText("{\"hello\": 1}" + " ".repeat(2048))
        File(legacyTokDir, "merges.txt").writeText("#version: 0.2\nh e\n" + " ".repeat(2048))

        // Re-initialize manager as happens on App Update / App Reopen
        val updatedManager = OnnxModelManager(context)
        val metrics = updatedManager.inspectCurrentMetrics()

        assertTrue(metrics.isTextEncoderReady)
        assertTrue(metrics.hasTokenizerFiles)
        assertTrue(updatedManager.isDedicatedSlotReady(DedicatedModelSlot.TEXT_ENCODER))
        assertTrue(updatedManager.isDedicatedSlotReady(DedicatedModelSlot.TOKENIZER))

        updatedManager.deleteAllModels()
    }

    @Test
    fun `core slot specs strictly include only 1A 1B 2A 2B 3 4 5 6 and prioritize Fast LCM first`() {
        val specs = OnnxAutoModelDownloader.CORE_SLOT_SPECS
        assertEquals(8, specs.size)
        assertTrue(specs.none { !it.slot.isCoreSlot })

        val displayCodes = specs.sortedBy { it.displayOrder }.map { it.shortCode }
        assertEquals(listOf("1A", "1B", "2A", "2B", "3", "4", "5", "6"), displayCodes)

        val executionCodes = specs.sortedBy { it.executionPriority }.map { it.shortCode }
        assertEquals(listOf("1A", "1B", "3", "4", "5", "6", "2A", "2B"), executionCodes)
    }

    @Test
    fun `downloader follows redirect resumes partial file and auto activates slot in IG_Models`() = runBlocking {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val manager = OnnxModelManager(context)
        manager.deleteAllModels()

        // Build a >600 KB valid ONNX payload for Slot 1A (minValidBytes = 500_000)
        val full1ABytes = buildValidOnnxHeaderPayload(size = 620_000, externalLocation = "model_int8.onnx.data")
        val vocabBytes = ("{\"<|startoftext|>\": 49406, \"<|endoftext|>\": 49407}" + " ".repeat(550_000))
            .toByteArray(Charsets.UTF_8)
        val mergesBytes = ("#version: 0.2\nt h\n" + " ".repeat(250_000))
            .toByteArray(Charsets.UTF_8)

        // Pre-seed a 200 KB .download_part file in Slot 1A to verify HTTP 206 Range resume!
        val fastDir = manager.targetDownloadFolder(DedicatedModelSlot.FAST_LCM_GRAPH)
        File(fastDir, "model.onnx.download_part").writeBytes(full1ABytes.copyOfRange(0, 200_000))

        var sawRangeResumeHeader = false

        val fakeDownloader = OnnxAutoModelDownloader(
            modelManager = manager,
            connectionFactory = { url ->
                val urlStr = url.toString()
                when {
                    urlStr.contains("huggingface.co") && urlStr.endsWith("/unet/model.onnx") -> {
                        FakeHttpConnection(
                            url = url,
                            statusCode = 302,
                            redirectLocation = "https://cdn-lfs.hf.co/fast_lcm/model.onnx",
                            fullPayload = ByteArray(0)
                        )
                    }
                    urlStr.contains("cdn-lfs.hf.co/fast_lcm/model.onnx") -> {
                        FakeHttpConnection(
                            url = url,
                            statusCode = 200,
                            fullPayload = full1ABytes,
                            onConnectCallback = { rangeHeader ->
                                if (rangeHeader == "bytes=200000-") {
                                    sawRangeResumeHeader = true
                                }
                            }
                        )
                    }
                    urlStr.endsWith("/tokenizer/vocab.json") -> {
                        FakeHttpConnection(url = url, statusCode = 200, fullPayload = vocabBytes)
                    }
                    urlStr.endsWith("/tokenizer/merges.txt") -> {
                        FakeHttpConnection(url = url, statusCode = 200, fullPayload = mergesBytes)
                    }
                    else -> error("Unexpected URL in test: $urlStr")
                }
            }
        )

        val completedCallbacks = mutableListOf<DedicatedModelSlot>()
        val batchRes = fakeDownloader.downloadSelectedCoreSlots(
            selectedSlots = setOf(DedicatedModelSlot.FAST_LCM_GRAPH, DedicatedModelSlot.TOKENIZER),
            onProgress = {},
            onSlotCompleted = { completedCallbacks.add(it) }
        ).getOrThrow()

        assertTrue("Expected Range: bytes=200000- header on redirected CDN request", sawRangeResumeHeader)
        assertEquals(
            listOf(DedicatedModelSlot.FAST_LCM_GRAPH, DedicatedModelSlot.TOKENIZER),
            batchRes.downloadedSlots
        )
        assertEquals(batchRes.downloadedSlots, completedCallbacks)
        assertFalse(File(fastDir, "model.onnx.download_part").exists())
        assertEquals(full1ABytes.size.toLong(), File(fastDir, "model.onnx").length())
        assertTrue(manager.isDedicatedSlotReady(DedicatedModelSlot.FAST_LCM_GRAPH))
        assertTrue(manager.isDedicatedSlotReady(DedicatedModelSlot.TOKENIZER))

        manager.deleteAllModels()
    }

    @Test
    fun `first launch dialog displays folder path all 8 core checkboxes and skip 2B preset`() {
        var confirmedSelection: Set<DedicatedModelSlot>? = null
        composeTestRule.setContent {
            MyApplicationTheme(darkTheme = false) {
                FirstLaunchModelDownloadDialog(
                    metrics = RealDeviceAndModelMetrics(
                        deviceFreeBytes = 24L * 1024L * 1024L * 1024L,
                        phoneModelsFolderPath = "Android/media/com.aistudio.imagegenerator.nvmkai/IG_Models"
                    ),
                    isTamil = true,
                    darkTheme = false,
                    onAllowAndDownload = { confirmedSelection = it },
                    onDismissLater = {}
                )
            }
        }
        composeTestRule.waitForIdle()

        composeTestRule.onNodeWithTag("first_launch_model_download_dialog").assertIsDisplayed()
        composeTestRule.onNodeWithTag("first_launch_folder_path_badge").performScrollTo().assertIsDisplayed()
        composeTestRule.onNodeWithTag("download_slot_checkbox_1a").performScrollTo().assertIsDisplayed()
        composeTestRule.onNodeWithTag("download_slot_checkbox_1b").performScrollTo().assertIsDisplayed()

        // Tap "Skip 2B" preset and then Allow & Download
        composeTestRule.onNodeWithTag("preset_skip_2b_heavy").performScrollTo().performClick()
        composeTestRule.waitForIdle()
        composeTestRule.onNodeWithTag("first_launch_allow_download_button").assertIsDisplayed().performClick()
        composeTestRule.waitForIdle()

        val selected = confirmedSelection ?: emptySet()
        assertEquals(7, selected.size)
        assertFalse(DedicatedModelSlot.STANDARD_SD_WEIGHTS in selected)
        assertTrue(DedicatedModelSlot.FAST_LCM_GRAPH in selected)
        assertTrue(DedicatedModelSlot.FAST_LCM_WEIGHTS in selected)
        assertTrue(DedicatedModelSlot.TOKENIZER in selected)
    }

    private fun buildValidOnnxHeaderPayload(size: Int, externalLocation: String? = null): ByteArray {
        val out = ByteArrayOutputStream()
        // Valid ONNX ModelProto: field 1 (ir_version) varint = 8 -> 0x08, 0x08
        out.write(byteArrayOf(0x08, 0x08))
        // field 2 (producer_name) string = "pytorch" -> 0x12, 0x07, "pytorch"
        val prod = "pytorch".toByteArray(Charsets.UTF_8)
        out.write(byteArrayOf(0x12, prod.size.toByte()))
        out.write(prod)
        if (externalLocation != null) {
            val locBytes = externalLocation.toByteArray(Charsets.UTF_8)
            out.write(byteArrayOf(0x0A, 0x08))
            out.write("location".toByteArray(Charsets.UTF_8))
            out.write(byteArrayOf(0x12, locBytes.size.toByte()))
            out.write(locBytes)
        }
        val current = out.size()
        if (size > current) {
            out.write(ByteArray(size - current))
        }
        return out.toByteArray()
    }

    private class FakeHttpConnection(
        url: URL,
        private val statusCode: Int,
        private val fullPayload: ByteArray,
        private val redirectLocation: String? = null,
        private val onConnectCallback: ((String?) -> Unit)? = null
    ) : HttpURLConnection(url) {
        private val reqHeaders = mutableMapOf<String, String>()
        private var effectiveSlice: ByteArray = fullPayload
        private var effectiveCode: Int = statusCode

        override fun setRequestProperty(key: String?, value: String?) {
            if (key != null && value != null) {
                reqHeaders[key] = value
            }
        }

        override fun connect() {
            connected = true
            val range = reqHeaders["Range"]
            onConnectCallback?.invoke(range)
            if (statusCode == 200 && range != null && range.startsWith("bytes=") && range.endsWith("-")) {
                val offset = range.removePrefix("bytes=").removeSuffix("-").toIntOrNull() ?: 0
                if (offset in 1 until fullPayload.size) {
                    effectiveCode = 206
                    effectiveSlice = fullPayload.copyOfRange(offset, fullPayload.size)
                }
            }
        }

        override fun getResponseCode(): Int {
            if (!connected) connect()
            return effectiveCode
        }

        override fun getHeaderField(name: String?): String? {
            if (name.equals("Location", ignoreCase = true)) return redirectLocation
            if (name.equals("Content-Length", ignoreCase = true)) return effectiveSlice.size.toString()
            return null
        }

        override fun getHeaderFieldLong(name: String?, Default: Long): Long {
            if (name.equals("Content-Length", ignoreCase = true)) return effectiveSlice.size.toLong()
            return Default
        }

        override fun getInputStream(): InputStream {
            if (!connected) connect()
            return ByteArrayInputStream(effectiveSlice)
        }

        override fun disconnect() {
            connected = false
        }

        override fun usingProxy(): Boolean = false
    }
}
