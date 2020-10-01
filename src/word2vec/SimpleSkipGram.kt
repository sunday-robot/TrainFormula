package word2vec

import nn.MatMulLayer
import nn.MatMulOneHotLayer
import nn.crossEntropyError
import nn.softMax
import java.io.ObjectInputStream
import java.io.ObjectOutputStream

class SimpleSkipGram(
        val inLayer: MatMulOneHotLayer,
        val outLayers: Array<MatMulLayer>
) {
    constructor(ois: ObjectInputStream) : this(
            ois.readObject() as MatMulOneHotLayer,
            Array<MatMulLayer>(ois.readInt()) {
                ois.readObject() as MatMulLayer
            })

    fun serialize(oos: ObjectOutputStream) {
        inLayer.serialize(oos)
        oos.writeInt(outLayers.count())
        outLayers.forEach {
            it.serialize(oos)
        }
    }

    /**
     * 個々のバッチの処理前に呼び、内部変数を初期化する。
     */
    fun reset() {
        inLayer.reset()
        outLayers.forEach { it.reset() }
    }

    fun predict(x: Int): List<Array<Float>> {
        val tmp = inLayer.evaluate(x)
        return List(outLayers.size) {
            val tmp1 = outLayers[it].evaluate(tmp)
            softMax(tmp1)
        }
    }

    fun loss(x: Int, t: List<Int>): Float {
        val y = predict(x)
        var l = 0f
        y.indices.forEach {
            l += crossEntropyError(y[it], t[it])
        }
        return l
    }

    fun gradient(t: List<Int>, x: Int) {
        val ilf = inLayer.forward(x)
        val olbSum = Array(inLayer.outputSize) { 0f }
        outLayers.indices.forEach { index ->
            val olf = softMax(outLayers[index].forward(ilf))
            // ロス値の計算はしなくてもdYは計算可能(それがsoftmaxを使用する理由にもなっているらしい)

            olf[t[index]] -= 1f
            val olb = outLayers[index].backward(olf)
            olbSum.indices.forEach {
                olbSum[it] += olb[it]
            }
        }
        inLayer.backward(olbSum)
    }

    fun wordVectorList(): Array<Array<Float>> {
        return Array(inLayer.inputSize) { i ->
            Array(inLayer.outputSize) { j ->
                inLayer.weight(i, j)
            }
        }
    }
}
