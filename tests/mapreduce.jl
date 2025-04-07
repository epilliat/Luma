include("../src/mapreduce.jl")

# Empty for now
using Test



@testset "MapReduce, *, +" begin
    mpr = MapReduce(storeGlmem=true)
    X = (1, 10, 100, 1000, 2000, 5000, 10000, 100000)
    n = length(X)
    result = CUDA.fill(0.0, n)
    l = []
    for (i, N) in enumerate(X)
        V = CUDA.rand(Float64, N)
        W = CUDA.ones(Float64, N)
        Vs = (V, W)
        push!(l, sum(V))
        mpr(*, +, view(result, i:i), Vs)
    end
    result_cpu = Array(result)
    @test all(isapprox.(l - result_cpu, 0; atol=1e-8))
end