#profile_ncu.jl
using CUDA

T = Float64
N = Int(1e6)
V = CuArray{T}(1:N |> collect)
w = CUDA.ones(T, N)
Vs = (V, w)
result = CuArray{T}([0.0]) # We store the result in the GPU
result_unified = CuArray{T,1,CUDA.UnifiedMemory}([0.0])
l = []
for _ in (1:1000)
    push!(l, CUDA.@sync CUDA.dot(V, w))
end
println(l)
