package floatArray

// a x b[]
operator fun Float.times(a: Array<Float>) = Array(a.size) { i ->
    this * a[i]
}
