include("../src/mapreduce.jl")


using BenchmarkTools

T = Float64
N = Int(1e6)
V = CuArray{T}(1:N |> collect)
w = CUDA.ones(T, N)
Vs = (V, w)
result = CuArray{T}([0.0]) # We store the result in the GPU
result_unified = CuArray{T,1,CUDA.UnifiedMemory}([0.0]) # Or in the CPU (less efficient)

# Sum of vector V

## CUDA.jl (mapreduce)

@btime CUDA.@sync sum($V) # 36 μs
x = CUDA.@bprofile time = 0.05 CUDA.@sync sum(V) # 36 μs
y = CUDA.@profile CUDA.@sync sum(V) # 36 μs
x.nvtx[1:4, :]
#%%
@btime CUDA.@sync mapreduce(identity, +, V) # 36.5 μs
@btime CUDA.@sync sum!($result, $V) # 37 μs (?!)

## Luma

#mpr = MapReduce(storeGlmem=true)
#mpr(identity, +, result, (V,)) # Kernel and Global memory are initialized at first run 
#@btime CUDA.@sync mpr(identity, +, result, (V,)) #24 μs
function tst()
    for _ in (1:1000)
        CUDA.@sync mpr(identity, +, result, (V,)) #24 μs
    end
    x = CUDA.@bprofile time = 0.05 mpr(identity, +, result, (V,)) #24 μs
end
#%%

# Dot Product

## CuBlas

@btime CUDA.@sync CUDA.dot($V, $w) # 36.5 μs
x = CUDA.@profile CUDA.@sync CUDA.dot(V, w)
CSV.write("./test_cublas.csv", x.device)

## Luma

Vs = (V, w)
mpr = MapReduce(storeGlmem=true)
mpr(*, +, result, Vs) # Kernel and Global memory are initialized at first run 
@btime CUDA.@sync mpr(*, +, $result, $Vs) # 27.3 μs (better than CuBlas)
names(x.device)
(x.device[:, "stop"] - x.device[:, "start"]) * 1e6
@show x.device
V
# Overhead

## If we do not store in global memory (we unsafe_free glmem at each run and reallocate)
#%%

Vs = (V, w)
mpr = MapReduce(storeGlmem=false)
mpr(*, +, result, Vs) # Kernel and Global memory are initialized at first run 
@btime CUDA.@sync mpr(*, +, $result, $Vs) # 32.1 μs (slightly better than CuBlas)

## If we relaunch kernel parameterization at each run
#%%

Vs = (V, w)
mpr = MapReduce()
mpr(*, +, result, Vs) # Kernel and Global memory are initialized at first run 
@btime CUDA.@sync mpr(*, +, $result, $Vs; reinit=true) # 32.45 μs (slightly better than CuBlas)

## If we do not relaunch kernel but send result to cpu through unified memory

Vs = (V, w)
mpr = MapReduce()
mpr(*, +, result_unified, Vs) # Kernel and Global memory are initialized at first run 
@btime CUDA.@sync mpr(*, +, $result_unified, $Vs; reinit=false) # 30.4 μs (slightly better than CuBlas)
