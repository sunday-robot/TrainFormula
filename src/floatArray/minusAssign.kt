package floatArray

// a[] -= b[]
operator fun Array<Float>.minusAssign(a: Array<Float>) {
    for (i in indices)
        this[i] -= a[i]
}
