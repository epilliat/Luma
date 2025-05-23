# Luma


This is an experimental Repo for CUDA parallel computing in Julia.

# Examples of usage

``` julia
N = Int(1e6)
V = CuArray{Float64}(1:N |> collect)
w = CUDA.ones(Float64, N)
Vs = (V,w) 
result = CuArray{Float64}([0.0])

mpr = MapReduce(storeGlmem=true) # If storeGlmem=true, we store in mpr two vectors of size of order the number of blocks of the kernel
mpr(identity, +, result, Vs) # Kernel and global memory are initialized at first run 
```

# Performances

See perf folder for more details.

## MapReduce, Sum of a Vector

Luma VS CUDA:

``` julia
@btime CUDA.@sync mpr(identity, +, $result, ($V,)) #(LUMA),24 μs
@btime CUDA.@sync sum!($result, $V) # (CUDA.jl) 37 μs
```

## Dot Product

Luma VS CuBlas:

``` julia
@btime CUDA.@sync mpr(*, +, $result, ($V, $w)) # (LUMA), 27.6 μs
@btime CUDA.@sync CUDA.dot($V, $w) # (CuBlas), 36.5 μs
```

## Accumulate

`V1` and `V2` are of size $10^6$ and $10^8$, respectively.

``` julia
### LUMA
@btime CUDA.@sync acc(identity, +, $R1, $V1) # 108.2 μs
@btime CUDA.@sync acc(identity, +, $R2, $V2) # 19 ms

### CUDA.jl
@btime CUDA.@sync accumulate!(+, R1, V1) # 301 μs
@btime CUDA.@sync accumulate!(+, R2, V2) # 35.1 ms
```
