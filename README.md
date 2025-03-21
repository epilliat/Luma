# Luma

This is an experimental Repo for CUDA parallel computing in Julia.

## Examples of usage

```julia
N = Int(1e6)
V = CuArray{Float64}(1:N |> collect)
w = CUDA.ones(Float64, N)
Vs = (V,w) 
result = CuArray{Float64}([0.0])

mpr = MapReduce(storeGlmem=true) # If storeGlmem=true, we store in mpr two vectors of size of order the number of blocks of the kernel
mpr(identity, +, result, Vs) # Kernel and global memory are initialized at first run 
```


## Performances

```julia
N = Int(1e6)
V = CuArray{Float64}(1:N |> collect)
w = CUDA.ones(Float64, N)
result = CuArray{Float64}([0.0]) # We store the result in the GPU
result_unified = CuArray{Float64,1,CUDA.UnifiedMemory}([0.0]) # Or in the CPU (less efficient)
```

# Sum of vector V

## CUDA.jl (mapreduce)

```julia
@btime CUDA.@sync sum($V) # 56.7 μs
@btime CUDA.@sync mapreduce(identity, +, V) # 54 μs
@btime CUDA.@sync sum!($result, $V) # 67.5 (?!)
```

## Luma

```julia
mpr = MapReduce(storeGlmem=true)
mpr(identity, +, result, Vs) # Kernel and Global memory are initialized at first run 
@btime CUDA.@sync mpr(identity, +, $result, $Vs) #27.8 μs
```


# Dot Product

## CuBlas

```julia
@btime CUDA.@sync CUDA.dot($V, $w) # 39.2 μs
```

## Luma

```julia
Vs = (V, w)
mpr = MapReduce(storeGlmem=true)
mpr(*, +, result, Vs) # Kernel and Global memory are initialized at first run 
@btime CUDA.@sync mpr(*, +, $result, $Vs) # 31.6 μs (slightly better than CuBlas) -- see the perf folder for a discussion on the overhead
```