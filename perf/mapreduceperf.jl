include("../src/mapreduce.jl")

# Update: Performances are much better with the linux open-source driver (at least +10% speed) than with proprietary.


using BenchmarkTools


N = Int(1e6)
V = CuArray{Float64}(1:N |> collect)
w = CUDA.ones(Float64, N)
Vs = (V, w)
result = CuArray{Float64}([0.0]) # We store the result in the GPU
result_unified = CuArray{Float64,1,CUDA.UnifiedMemory}([0.0]) # Or in the CPU (less efficient)

# Sum of vector V

## CUDA.jl (mapreduce)

@btime CUDA.@sync sum($V) # 36 μs
@btime CUDA.@sync mapreduce(identity, +, V) # 36.5 μs
@btime CUDA.@sync sum!($result, $V) # 37 μs (?!)

## Luma

mpr = MapReduce(storeGlmem=true)
mpr(identity, +, result, (V,)) # Kernel and Global memory are initialized at first run 
@btime CUDA.@sync mpr(identity, +, $result, ($V,)) #24 μs



# Dot Product

## CuBlas

@btime CUDA.@sync CUDA.dot($V, $w) # 36.5 μs

## Luma

Vs = (V, w)
mpr = MapReduce(storeGlmem=true)
mpr(*, +, result, Vs) # Kernel and Global memory are initialized at first run 
@btime CUDA.@sync mpr(*, +, $result, $Vs) # 27.6 μs (slightly better than CuBlas)



# Overhead

## If we do not store in global memory (we unsafe_free glmem at each run and reallocate)
#%%

Vs = (V, w)
mpr = MapReduce(storeGlmem=false)
mpr(*, +, result, Vs) # Kernel and Global memory are initialized at first run 
@btime CUDA.@sync mpr(*, +, $result, $Vs) # 37.1 μs (slightly better than CuBlas)

## If we relaunch kernel parameterization at each run
#%%

Vs = (V, w)
mpr = MapReduce()
mpr(*, +, result, Vs) # Kernel and Global memory are initialized at first run 
@btime CUDA.@sync mpr(*, +, $result, $Vs; reinit=true) # 40.5 μs (slightly worse than CuBlas)

## If we do not relaunch kernel but send result to cpu through unified memory

Vs = (V, w)
mpr = MapReduce()
mpr(*, +, result_unified, Vs) # Kernel and Global memory are initialized at first run 
@btime CUDA.@sync mpr(*, +, $result_unified, $Vs; reinit=false) # 39.7 μs (slightly worse than CuBlas)
