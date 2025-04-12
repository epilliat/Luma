include("../src/accumulate.jl")

using BenchmarkTools


N1 = Int(1e6)
N2 = Int(1e8)
V1 = CuArray{Float64}(1:N1 |> collect)
V2 = CuArray{Float64}(1:N2 |> collect)
R1 = CUDA.fill(0.0, N1) # We store the result in the GPU
R2 = CUDA.fill(0.0, N2)

## CUDA.jl

@btime CUDA.@sync accumulate!(+, R1, V1) # 301 μs
@btime CUDA.@sync accumulate!(+, R2, V2) # 35.1 ms

## Luma

acc = Accumulate(storeGlmem=true)

acc(identity, +, R1, V1)
R1
@btime CUDA.@sync acc(identity, +, $R1, $V1) # 108.2 μs (∼ 3x faster than CUDA.jl)

acc(identity, +, R2, V2)
@btime CUDA.@sync acc(identity, +, $R2, $V2) # 19 ms (almost 2x faster than CUDA.jl)