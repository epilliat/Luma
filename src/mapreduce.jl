include("../helpers.jl")
include("kernels/mapreducekernel.jl")


struct MapReduceConfig{F<:Function,O<:Function,T}
    kernel::CUDA.HostKernel
    config::@NamedTuple{blocks::Int, threads::Int} #Should be the config given by the kernel
    shmemsize::Int
    f::F
    op::O
    lengthVs::Int
    source::T
    #out::Outf # Example of out type of f: T -> Outf.
end

struct MapReduceGlmem{Outf}
    partial::AbstractGPUVector{Outf}
    flag::AbstractGPUVector{FLAG_TYPE}
end

"""
    MapReduce

A GPU-accelerated map-reduce framework using CUDA for parallel computation. 
We initialize a callable object mpr::MapReduce whose operators f, op can dynamically change.
It also generalizes coordinate-wise dot product.

# Fields
- `config::Union{MapReduceConfig,Nothing}`: Configuration details about the Kernel, its config, operations and datatype
- `glmem::Union{MapReduceGlmem,Nothing}`: Global memory on the GPU of order number of blocks
- `storeGlmem::Bool`: Whether to retain global memory allocations between operations

# Example
```julia
N = Int(1e6)
V = CuArray{Float64}(1:N |> collect)
w = CUDA.ones(Float64, N)
Vs = (V,w) 
result = CuArray{Float64}([0.0])

mpr = MapReduce(storeGlmem=true) # If storeGlmem=true, we store in mpr two vectors of size of order the number of blocks of the kernel
mpr(identity, +, result, Vs) # Kernel and global memory are initialized at first run 
```
"""
mutable struct MapReduce
    config::Union{MapReduceConfig,Nothing}
    glmem::Union{MapReduceGlmem,Nothing}
    storeGlmem::Bool
end

MapReduce(; storeGlmem=true) = MapReduce(nothing, nothing, storeGlmem)

#Warning: result must have right eltype Outf
function (mpr::MapReduce)(f::F, op::O, result::AbstractGPUVector{Outf}, Vs::NTuple{K,AbstractGPUVector{T}}; reinit=false) where {T,Outf,K,F<:Function,O<:Function}
    reinit = reinit || mpr.config === nothing || T !== typeof(mpr.config.source) || typeof(mpr.config.f) !== F || typeof(mpr.config.op) !== O || K != mpr.config.lengthVs
    if reinit
        bytes = zeros(UInt8, sizeof(T))
        source = reinterpret(T, bytes)[1]
        #out = f((source for _ in (1:K))...) # Example of an element of right type
        #PARTIAL = CuArray{typeof(out)}(undef, 0) # Example of cuArray of right type. We take the cuarray result instead
        kernel = @cuda launch = false mapreducekernel(f, op, result, Vs, result, FLAG_AR, FLAG_TYPE(0), 200)
        config = launch_configuration(kernel.fun; shmem=(threads) -> 32 * sizeof(Outf))
        mpr.config = MapReduceConfig(
            kernel,
            config,
            32 * sizeof(Outf),
            f, op,
            K,
            source
        )
    end

    if reinit || (mpr.glmem === nothing) # Global memory allocation
        glmemlength = total_glmem_length(Val(mpr.config.config.blocks), Val(32)) # ***the actual number of blocks glmemlength used in kernel can be smaller
        mpr.glmem = MapReduceGlmem(
            CuArray{Outf}(undef, glmemlength),
            CuArray{FLAG_TYPE}(undef, glmemlength) # Allocating two cuArray is costly, (∼ 1.5μs). Can we do this in one go ? The pb is they have not same type
        )
    end

    N = length(Vs[1])
    threads = min(mpr.config.config.threads, N)
    blocks = min(mpr.config.config.blocks, cld(N, threads))
    targetflag = rand(FLAG_TYPE)
    glmemlength = total_glmem_length(Val(blocks), Val(32)) # ***
    (mpr.config.kernel)(
        mpr.config.f, mpr.config.op,
        result,
        Vs,
        mpr.glmem.partial,
        mpr.glmem.flag,
        targetflag,
        glmemlength;
        shmem=mpr.config.shmemsize,
        threads=threads,
        blocks=blocks
    )

    if !mpr.storeGlmem
        CUDA.unsafe_free!(mpr.glmem.partial)
        CUDA.unsafe_free!(mpr.glmem.flag)
        mpr.glmem = nothing
    end
end


