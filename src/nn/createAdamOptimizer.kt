package nn

fun createAdamOptimizer(
        parameterCount: Int, // 重み値の個数
        learningRate: Float = 0.001f,
        beta1: Float = 0.9f,
        beta2: Float = 0.999f
) = AdamOptimizer(
        learningRate, beta1, beta2, 0,
        MutableList(parameterCount) { 0f },
        MutableList(parameterCount) { 0f })
