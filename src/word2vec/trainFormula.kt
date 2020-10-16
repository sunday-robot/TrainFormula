package word2vec

import common.log
import floatArray.printArray
import nn.createShuffledIndices
import nn.random.reset
import kotlin.math.max
import kotlin.math.min

/**
 * word2vecの理解を目的としたプログラム。
 * 「-1+3=2」のような二つの足し算の計算式を文、登場する数値を単語として登場する数値の単語ベクトルを求める。
 */
fun main() {
    // 語彙の情報(本プログラムでは、minValue以上maxValue以下の整数値が語彙である)
    val minValue = -9
    val maxValue = 9
    val vocabularySize = -minValue + 1 + maxValue

    // word2vecのネットワークの設定
    val wordVectorSize = 2  // 単語ベクトルの次元数(本家word2vecのデフォルト値は200次元とのこと)
    val windowSize = 1    // コンテキストのサイズ(の半分)。本プログラムでは計算式の左辺に登場する二つの数値をコンテキストとしているので、1

    // 学習パラメータ
    val batchSize = 10  // 学習時のバッチサイズ
    val epochCount = 10000  // エポック数

    // 学習データとword2vecのネットワークを生成し、学習を行う。
    val trainingData = createTrainingData(minValue, maxValue)
    val network = createNetwork(vocabularySize, wordVectorSize, windowSize)
    trainNetwork(network, trainingData, epochCount, batchSize)

    // (debug)出来上がったNNで、推論を行う。
    for (i in minValue..maxValue) {
        val p = network.predict(i - minValue)
        println("${i}->")
        for (c in p) {
            printArray(c)
        }
        println("")
    }

    // word2vecのニューラルネットワークの第１レイヤーの重み値を単語ベクトルとして取り出す。
    val wordVectorList = network.wordVectorList()

    // 単語ベクトルの値をコンソールに出力する。(Excelに取り込みやすいように、TSV形式で出力する)
    for (word in minValue..maxValue) {
        print("$word")
        for (e in wordVectorList[word - minValue])
            print("\t$e")
        println()
    }
}

/**
 * 学習データ(ターゲットとコンテキストのリスト)を作成する。
 * "-1+3=2"の式の場合、2がターゲットで、-1と3がコンテキスト。
 */
fun createTrainingData(minValue: Int, maxValue: Int): MutableList<TargetAndContext> {
    val targetAndContextList = mutableListOf<TargetAndContext>()
    for (context1 in minValue..maxValue)
        for (context2 in max(minValue - context1, minValue)..min(maxValue - context1, maxValue)) {
            val target = context1 + context2
            targetAndContextList.add(
                    TargetAndContext(
                            target - minValue, listOf(context1 - minValue, context2 - minValue)
                    )
            )
        }
    return targetAndContextList
}

/**
 * word2vecのニューラルネットワークを生成する。
 */
fun createNetwork(vocabularySize: Int, wordVectorSize: Int, windowSize: Int): SimpleSkipGram {
    reset(1L)
    return createSimpleSkipGram(vocabularySize, wordVectorSize, windowSize)
}

/**
 * word2vecのニューラルネットワークの学習を行う。
 */
fun trainNetwork(network: SimpleSkipGram, trainingData: MutableList<TargetAndContext>, epochCount: Int, batchSize: Int) {
    val vocabularySize = network.inLayer.inputSize
    val wordVectorSize = network.inLayer.outputSize
    val windowSize = network.outLayers.size / 2

    reset(1L)
    val optimizer = createSimpleSkipGramOptimizer(vocabularySize, wordVectorSize, windowSize)

    for (i in 0.until(epochCount)) {    // エポック数分のループ
        val trainingDataIndices = createShuffledIndices(trainingData.size) // 学習データのインデックス値をランダムに並べたリスト
        for (j in 0.until(trainingData.size) step batchSize) {
//            log("${i}/${epochCount} - ${j}/${targetAndContextList.size}: initializing weight gradients.")
            network.reset() // ネットワークの重み値の微分値の累積値の０クリア
            val bs = min(batchSize, trainingData.size - j)
            for (k in 0.until(bs)) {
                val idx = trainingDataIndices[j + k]
                val tc = trainingData[idx]
//                log("${i}/${epochCount} - ${j + k}/${targetAndContextList.size}: calculating weight gradients.")
                network.gradient(tc.context, tc.target)   // 重み値の微分値を求め、累積する
            }
//            log("${i}/${epochCount} - ${j}/${targetAndContextList.size}: optimizing weights.")
            optimizer.update(network)   // 重み値の微分値の累積値に従い、重み値を更新する
        }
        var loss = 0f
        trainingData.indices.forEach {
            val tc = trainingData[it]
            loss += network.loss(tc.target, tc.context)
        }
        log("$i: loss = $loss")
    }
}
