package com.example

import com.example.ai.ConvRepairMode
import com.example.ai.DiffusionModelSlot
import com.example.ai.OnnxModelManager
import com.example.ai.OnnxProtoCodec
import com.example.ai.OnnxProtoCodec.DT_FLOAT
import com.example.ai.OnnxProtoCodec.DT_INT8
import com.example.ai.OnnxProtoCodec.DT_UINT8
import com.example.ai.OnnxProtoCodec.OnnxExternalEntry
import com.example.ai.OnnxProtoCodec.OnnxGraphProto
import com.example.ai.OnnxProtoCodec.OnnxModelProto
import com.example.ai.OnnxProtoCodec.OnnxNodeProto
import com.example.ai.OnnxProtoCodec.OnnxOpsetProto
import com.example.ai.OnnxProtoCodec.OnnxTensorProto
import com.example.ai.OnnxProtoCodec.OnnxValueInfoProto
import com.example.ai.OnnxQuantGraphRepairer
import com.example.ai.PreferredUnetImportMode
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.nio.file.Files
import kotlin.math.abs

class OnnxQuantGraphRepairerTest {

    private fun makeValueInfo(name: String): OnnxValueInfoProto {
        val w = OnnxProtoCodec.ProtoWriter(32)
        w.writeStringField(1, name)
        return OnnxValueInfoProto(name, w.toByteArray())
    }

    @Test
    fun cleanModelWithoutConvInteger_isMarkedCleanWithoutModifyingBytes() {
        val dir = Files.createTempDirectory("clean_onnx_").toFile()
        try {
            val graph = OnnxGraphProto(
                name = "clean_graph",
                nodes = mutableListOf(
                    OnnxNodeProto(
                        inputs = mutableListOf("x", "w"),
                        outputs = mutableListOf("y"),
                        name = "/conv_in/Conv",
                        opType = "Conv"
                    )
                ),
                initializers = mutableListOf(
                    OnnxProtoCodec.createFloatInitializer("w", longArrayOf(1, 1, 3, 3), FloatArray(9) { 0.5f })
                ),
                inputs = mutableListOf(makeValueInfo("x")),
                outputs = mutableListOf(makeValueInfo("y"))
            )
            val model = OnnxModelProto(
                irVersion = 8L,
                opsets = mutableListOf(OnnxOpsetProto("", 15L)),
                graph = graph
            )
            val modelFile = File(dir, "model.onnx")
            val originalBytes = OnnxProtoCodec.serializeModel(model)
            modelFile.writeBytes(originalBytes)

            val mode = OnnxQuantGraphRepairer.ensureCompatibleBeforeSession(modelFile)
            assertEquals(ConvRepairMode.CLEAN, mode)
            assertArrayEquals(originalBytes, modelFile.readBytes())
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun tier1Repair_convertsExternalSignedInt8ConvIntegerToUint8_preservingMatMulAndExactMath() {
        val dir = Files.createTempDirectory("lcm_int8_tier1_").toFile()
        try {
            // External companion file: [0..63] = MatMul INT8 weights (must NOT change),
            // [64..79] = ConvInteger INT8 weights (16 signed bytes), [80..127] = trailing data (must NOT change)
            val extName = "model_int8.onnx.data"
            val extFile = File(dir, extName)
            val fullExtBytes = ByteArray(128) { i -> (i - 50).toByte() }
            val originalConvWeights = byteArrayOf(
                -128, -100, -55, -12, -1, 0, 1, 7, 19, 42, 64, 88, 101, 115, 126, 127
            )
            System.arraycopy(originalConvWeights, 0, fullExtBytes, 64, 16)
            val untouchedPrefix = fullExtBytes.copyOfRange(0, 64)
            val untouchedSuffix = fullExtBytes.copyOfRange(80, 128)
            extFile.writeBytes(fullExtBytes)

            val convWeightTensor = OnnxTensorProto(
                dims = mutableListOf(2L, 2L, 2L, 2L),
                dataType = DT_INT8,
                name = "unet.conv_in.weight_quantized",
                externalData = mutableListOf(
                    OnnxExternalEntry("location", extName),
                    OnnxExternalEntry("offset", "64"),
                    OnnxExternalEntry("length", "16")
                ),
                dataLocation = OnnxProtoCodec.DATA_LOC_EXTERNAL
            )
            // Shared INT8 zero-point = -5 used by BOTH ConvInteger AND MatMulInteger!
            val sharedS8Zp = OnnxProtoCodec.create8BitInitializer(
                name = "shared_int8_zp",
                dataType = DT_INT8,
                dims = longArrayOf(),
                values = byteArrayOf((-5).toByte())
            )
            val convWScale = OnnxProtoCodec.createFloatInitializer(
                name = "conv_w_scale",
                dims = longArrayOf(),
                values = floatArrayOf(0.025f)
            )
            val biasInit = OnnxProtoCodec.createFloatInitializer(
                name = "conv_bias",
                dims = longArrayOf(1, 2, 1, 1),
                values = floatArrayOf(0.1f, -0.2f)
            )
            val matmulWeight = OnnxTensorProto(
                dims = mutableListOf(8L, 8L),
                dataType = DT_INT8,
                name = "attn.to_q.weight_quantized",
                externalData = mutableListOf(
                    OnnxExternalEntry("location", extName),
                    OnnxExternalEntry("offset", "0"),
                    OnnxExternalEntry("length", "64")
                ),
                dataLocation = OnnxProtoCodec.DATA_LOC_EXTERNAL
            )

            val convAttrPads = OnnxProtoCodec.buildIntAttributeRaw("group", 1L)
            val nodes = mutableListOf(
                OnnxNodeProto(
                    inputs = mutableListOf("sample"),
                    outputs = mutableListOf("x_q", "x_scale", "x_zp"),
                    name = "/conv_in/Conv_dql",
                    opType = "DynamicQuantizeLinear"
                ),
                OnnxNodeProto(
                    inputs = mutableListOf("x_q", "unet.conv_in.weight_quantized", "x_zp", "shared_int8_zp"),
                    outputs = mutableListOf("conv_int32"),
                    name = "/conv_in/Conv_quant",
                    opType = "ConvInteger",
                    attributesRaw = mutableListOf(convAttrPads)
                ),
                OnnxNodeProto(
                    inputs = mutableListOf("conv_int32"),
                    outputs = mutableListOf("conv_fp32"),
                    name = "/conv_in/Conv_cast",
                    opType = "Cast"
                ),
                OnnxNodeProto(
                    inputs = mutableListOf("x_scale", "conv_w_scale"),
                    outputs = mutableListOf("combined_scale"),
                    name = "/conv_in/Conv_scale_mul",
                    opType = "Mul"
                ),
                OnnxNodeProto(
                    inputs = mutableListOf("conv_fp32", "combined_scale"),
                    outputs = mutableListOf("conv_scaled"),
                    name = "/conv_in/Conv_out_mul",
                    opType = "Mul"
                ),
                OnnxNodeProto(
                    inputs = mutableListOf("conv_scaled", "conv_bias"),
                    outputs = mutableListOf("conv_out"),
                    name = "/conv_in/Conv_bias_add",
                    opType = "Add"
                ),
                // MatMulInteger sharing x_q and shared_int8_zp
                OnnxNodeProto(
                    inputs = mutableListOf("x_q", "attn.to_q.weight_quantized", "x_zp", "shared_int8_zp"),
                    outputs = mutableListOf("mm_out"),
                    name = "/attn/MatMul_quant",
                    opType = "MatMulInteger"
                )
            )

            val model = OnnxModelProto(
                irVersion = 8L,
                opsets = mutableListOf(OnnxOpsetProto("", 14L)),
                graph = OnnxGraphProto(
                    name = "lcm_unet",
                    nodes = nodes,
                    initializers = mutableListOf(convWeightTensor, sharedS8Zp, convWScale, biasInit, matmulWeight),
                    inputs = mutableListOf(makeValueInfo("sample")),
                    outputs = mutableListOf(makeValueInfo("conv_out"), makeValueInfo("mm_out"))
                )
            )
            val modelFile = File(dir, "model.onnx")
            modelFile.writeBytes(OnnxProtoCodec.serializeModel(model))

            assertTrue(OnnxQuantGraphRepairer.containsConvIntegerOp(modelFile))

            var statusReported = false
            val repairMode = OnnxQuantGraphRepairer.ensureCompatibleBeforeSession(modelFile) {
                statusReported = true
            }
            assertEquals(ConvRepairMode.TIER1_UINT8_CONV_INTEGER, repairMode)
            assertTrue(statusReported)

            // Verify external companion location discovery still works for OnnxModelManager!
            assertEquals(listOf(extName), OnnxModelManager.extractExternalDataLocations(modelFile))

            // Verify prefix & suffix of external file were NOT touched
            val patchedExtBytes = extFile.readBytes()
            assertArrayEquals(untouchedPrefix, patchedExtBytes.copyOfRange(0, 64))
            assertArrayEquals(untouchedSuffix, patchedExtBytes.copyOfRange(80, 128))

            // Verify parsed Tier-1 graph and exact integer math equivalence
            val parsedTier1 = OnnxProtoCodec.parseModel(modelFile.readBytes())
            val g1 = parsedTier1.graph!!
            val init1 = g1.initializers.associateBy { it.name }

            val patchedConvW = init1["unet.conv_in.weight_quantized"]!!
            assertEquals(DT_UINT8, patchedConvW.dataType)

            // Shared INT8 zp must remain DT_INT8 (-5) for MatMulInteger!
            val untouchedSharedZp = init1["shared_int8_zp"]!!
            assertEquals(DT_INT8, untouchedSharedZp.dataType)
            assertEquals(-5, OnnxProtoCodec.readTensor8BitBytes(untouchedSharedZp, dir)!![0].toInt())

            val convNode1 = g1.nodes.first { it.name == "/conv_in/Conv_quant" }
            val newU8ZpInit = init1[convNode1.inputs[3]]!!
            assertEquals(DT_UINT8, newU8ZpInit.dataType)
            val u8Zp = OnnxProtoCodec.readTensor8BitBytes(newU8ZpInit, dir)!![0].toInt() and 0xFF
            assertEquals(123, u8Zp) // -5 + 128 = 123

            val u8WeightBytes = OnnxProtoCodec.readTensor8BitBytes(patchedConvW, dir)!!
            val xU8 = 200
            val xZp = 110
            for (i in originalConvWeights.indices) {
                val wS8 = originalConvWeights[i].toInt() // signed [-128..127]
                val wU8 = u8WeightBytes[i].toInt() and 0xFF // unsigned [0..255]
                val origProd = (xU8 - xZp) * (wS8 - (-5))
                val newProd = (xU8 - xZp) * (wU8 - u8Zp)
                assertEquals("Exact int32 product mismatch at index $i", origProd, newProd)
            }

            // Calling a second time must hit marker cache and NOT flip external bytes again!
            var secondStatus = false
            val secondMode = OnnxQuantGraphRepairer.ensureCompatibleBeforeSession(modelFile) {
                secondStatus = true
            }
            assertEquals(ConvRepairMode.TIER1_UINT8_CONV_INTEGER, secondMode)
            assertFalse(secondStatus)
            assertArrayEquals(patchedExtBytes, extFile.readBytes())

            // Now test seamless fallback upgrade from Tier-1 UINT8 -> Tier-2 QDQ (DequantizeLinear + Conv)!
            assertTrue(OnnxQuantGraphRepairer.upgradeToQdqConv(modelFile))
            assertFalse(OnnxQuantGraphRepairer.containsConvIntegerOp(modelFile))
            assertEquals(ConvRepairMode.TIER2_QDQ_CONV, OnnxQuantGraphRepairer.getRecordedRepairMode(modelFile))

            val parsedTier2 = OnnxProtoCodec.parseModel(modelFile.readBytes())
            val g2 = parsedTier2.graph!!
            val opTypes2 = g2.nodes.map { it.opType }
            assertFalse("ConvInteger" in opTypes2)
            assertTrue("DequantizeLinear" in opTypes2)
            assertTrue("Conv" in opTypes2)
            assertTrue("Add" in opTypes2) // Downstream bias Add preserved!
            // DynamicQuantizeLinear must STAY alive because MatMulInteger still consumes x_q!
            assertTrue("DynamicQuantizeLinear" in opTypes2)
            assertTrue("MatMulInteger" in opTypes2)

            // Verify DequantizeLinear produces exact same float weights from the Tier-1 UINT8 tensor
            val dqNode = g2.nodes.first { it.opType == "DequantizeLinear" }
            val init2 = g2.initializers.associateBy { it.name }
            val dqScale = OnnxProtoCodec.readTensorFloats(init2[dqNode.inputs[1]]!!, dir)!![0]
            val dqZp = OnnxProtoCodec.readTensor8BitBytes(init2[dqNode.inputs[2]]!!, dir)!![0].toInt() and 0xFF
            for (i in originalConvWeights.indices) {
                val expectedFp = (originalConvWeights[i].toInt() - (-5)) * 0.025f
                val actualFp = ((u8WeightBytes[i].toInt() and 0xFF) - dqZp) * dqScale
                assertTrue(abs(expectedFp - actualFp) < 1e-6f)
            }
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun tier2Repair_rewritesPerChannelConvIntegerDirectlyAndPrunesDeadDynamicQuantizeLinear() {
        val dir = Files.createTempDirectory("lcm_per_channel_").toFile()
        try {
            // 3 output channels -> per-channel 1D w_zp [3] and 3D broadcast w_scale [3, 1, 1]
            val wInit = OnnxProtoCodec.create8BitInitializer(
                name = "conv_w_q",
                dataType = DT_INT8,
                dims = longArrayOf(3, 4, 3, 3),
                values = ByteArray(3 * 4 * 3 * 3) { (it - 40).toByte() }
            )
            val wScale3d = OnnxProtoCodec.createFloatInitializer(
                name = "conv_w_scale_3d",
                dims = longArrayOf(3, 1, 1),
                values = floatArrayOf(0.01f, 0.02f, 0.03f)
            )
            val wZp1d = OnnxProtoCodec.create8BitInitializer(
                name = "conv_w_zp_1d",
                dataType = DT_INT8,
                dims = longArrayOf(3),
                values = byteArrayOf(-2, 0, 4)
            )

            val nodes = mutableListOf(
                OnnxNodeProto(
                    inputs = mutableListOf("x_in"),
                    outputs = mutableListOf("x_q", "x_s", "x_zp"),
                    name = "dql_0",
                    opType = "DynamicQuantizeLinear"
                ),
                OnnxNodeProto(
                    inputs = mutableListOf("x_q", "conv_w_q", "x_zp", "conv_w_zp_1d"),
                    outputs = mutableListOf("y_i32"),
                    name = "/down_blocks.0/Conv_quant",
                    opType = "ConvInteger"
                ),
                OnnxNodeProto(
                    inputs = mutableListOf("y_i32"),
                    outputs = mutableListOf("y_f32"),
                    name = "cast_0",
                    opType = "Cast"
                ),
                OnnxNodeProto(
                    inputs = mutableListOf("x_s", "conv_w_scale_3d"),
                    outputs = mutableListOf("prod_scale"),
                    name = "scale_mul_0",
                    opType = "Mul"
                ),
                OnnxNodeProto(
                    inputs = mutableListOf("y_f32", "prod_scale"),
                    outputs = mutableListOf("y_out"),
                    name = "out_mul_0",
                    opType = "Mul"
                )
            )

            val model = OnnxModelProto(
                irVersion = 7L,
                opsets = mutableListOf(OnnxOpsetProto("", 11L)),
                graph = OnnxGraphProto(
                    name = "per_channel_unet",
                    nodes = nodes,
                    initializers = mutableListOf(wInit, wScale3d, wZp1d),
                    inputs = mutableListOf(makeValueInfo("x_in")),
                    outputs = mutableListOf(makeValueInfo("y_out"))
                )
            )
            val modelFile = File(dir, "model.onnx")
            modelFile.writeBytes(OnnxProtoCodec.serializeModel(model))

            val mode = OnnxQuantGraphRepairer.ensureCompatibleBeforeSession(modelFile)
            assertEquals(ConvRepairMode.TIER2_QDQ_CONV, mode)

            val repaired = OnnxProtoCodec.parseModel(modelFile.readBytes())
            // Opset upgraded from 11 -> 13 for per-channel DequantizeLinear(axis=0)
            assertEquals(13L, repaired.opsets.first().version)

            val rg = repaired.graph!!
            // Only DequantizeLinear -> Conv should remain; unshared DynamicQuantizeLinear, Cast, Muls pruned!
            assertEquals(listOf("DequantizeLinear", "Conv"), rg.nodes.map { it.opType })

            val dqNode = rg.nodes[0]
            val convNode = rg.nodes[1]
            assertEquals(1, dqNode.attributesRaw.size) // axis=0 attribute present
            assertEquals(listOf("x_in", dqNode.outputs[0]), convNode.inputs)
            assertEquals(listOf("y_out"), convNode.outputs)

            val rInit = rg.initializers.associateBy { it.name }
            val dqScaleInit = rInit[dqNode.inputs[1]]!!
            val dqZpInit = rInit[dqNode.inputs[2]]!!
            assertEquals(DT_FLOAT, dqScaleInit.dataType)
            assertEquals(listOf(3L), dqScaleInit.dims)
            assertEquals(DT_INT8, dqZpInit.dataType)
            assertEquals(listOf(3L), dqZpInit.dims)
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun folderResolver_prioritizesModelInt8OnnxDataInFlatUserFolder() {
        val parent = "image generator"
        val candidates = listOf(
            OnnxModelManager.CandidateModelFile(0, "text_encoder.onnx", parent, 125_000_000L),
            OnnxModelManager.CandidateModelFile(1, "unet.onnx", parent, 1_750_000L),
            OnnxModelManager.CandidateModelFile(2, "model.onnx", parent, 3_198_000L),
            OnnxModelManager.CandidateModelFile(3, "vae_decoder.onnx", parent, 198_000_000L)
        )
        val siblings = mapOf(
            parent to listOf(
                "text_encoder.onnx",
                "unet.onnx",
                "weights.pb",
                "model.onnx",
                "model_int8.onnx.data",
                "vae_decoder.onnx"
            )
        )
        val sizes = mapOf(
            parent to mapOf(
                "weights.pb" to 3_438_000_000L,
                "model_int8.onnx.data" to 902_750_000L
            )
        )

        val resolved = OnnxModelManager.resolveFolderSlotMatches(
            candidates = candidates,
            allFileNamesByParent = siblings,
            preferredUnetMode = PreferredUnetImportMode.AUTO_FAST_MOBILE,
            companionSizesByParent = sizes
        )
        assertEquals(2, resolved[DiffusionModelSlot.UNET])
        assertTrue(
            OnnxModelManager.shouldCopyCompanionWeight(
                onnxFileName = "model.onnx",
                onnxSizeBytes = 3_198_000L,
                companionFileName = "model_int8.onnx.data"
            )
        )
    }
}
