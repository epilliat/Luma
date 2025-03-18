include("../helpers.jl")



using CUDA, Random
import CUDA: AbstractGPUVector

const FLAG_TYPE = UInt64 # Risk of error of order 1/2^64
const FLAG = CuArray{FLAG_TYPE,1,CUDA.DeviceMemory}(undef, 0) # This is to determine the parameters of the kernel without redefining a FLAG CuArray each time.


@inline function strided_sum(f, op, Vs, idx, strd, n, ::Val{1})
    @inbounds val = f(Vs[1][idx])
    @inbounds for i in idx+strd:strd:n
        val = op(val, f(Vs[1][i]))#
    end

    return val
end
@inline function strided_sum(f, op, Vs, idx, strd, n, ::Val{2})
    @inbounds val = f(Vs[1][idx], Vs[2][idx])
    @inbounds for i in idx+strd:strd:n
        val = op(val, f(Vs[1][i], Vs[2][i]))#
        #
    end
    return val
end
@inline function strided_sum(f, op, Vs, idx, strd, n, ::Val{K}) where {K}
    @inbounds val = op(val, f((V[idx] for V in Vs)...))# Somehow less optimized than two previous functions
    @inbounds for i in idx+strd:strd:n
        val = op(val, f((V[i] for V in Vs)...))#
    end
    return val
end


function mapreducekernel(
    f, op,
    result,
    Vs::NTuple{K,AbstractVector},
    partial,
    flag,
    targetflag,
    glmemlength
) where {K}
    n = length(Vs[1])
    blocks = gridDim().x
    threads = UInt32(blockDim().x)
    warpsz = UInt32(warpsize())
    bidx = UInt32(blockIdx().x)
    tidx = UInt32(threadIdx().x)
    strd = threads * blocks
    idx = (bidx - 1) * threads + tidx

    idx > n && return


    warpsz = 32
    #@cushow cld(1024, 32), Int(cld(threads, 32))
    warps = (blocks * threads) ÷ warpsz
    lidx = ((tidx - 1) % warpsz + 1) #lane
    widx = (tidx - lidx) ÷ warpsz + 1 #warp index in the block
    gwidx = (idx - 1) ÷ warpsz + 1 # global warp index
    gridwidx = (gwidx - 1) * warpsz + 1 # first index of the warp in the grid

    cpt = 1

    shmem_res = @cuDynamicSharedMem(Float64, 32)

    val = strided_sum(f, op, Vs, idx, strd, n, Val(K))

    for u in powers_of_two(Val(warpsz))
        shuffled = shfl_up_sync(0xffffffff, val, u)
        if lidx > u
            val = op(val, shuffled)
        end
    end

    if lidx == warpsz || gridwidx + lidx - 1 == n
        @inbounds shmem_res[widx] = val
    end

    sync_threads()

    widx != 1 && return # We keep the first warp of the thread

    val = shmem_res[lidx]
    for u in powers_of_two(Val(warpsz))
        shuffled = shfl_up_sync(0xffffffff, val, u)
        if lidx > u
            val = op(val, shuffled)
        end
    end
    if lidx == cld(threads, warpsz)
        @inbounds partial[bidx] = val
    end
    threadfence()
    flag[bidx] = targetflag

    (widx - 1) != 0 && return # We keep only the first warp of each block (which contains 32 warps). It now shall cover [(bidx-1)*warpsz +1; min(bidx*warpsz, blocks)]

    curlength = blocks # current number of warps
    curwidx = cld(bidx, warpsz) # current index of the warp
    #

    shift = 0
    @inbounds while curlength != 1
        (curwidx - 1) * warpsz + lidx > curlength && return

        while true #Wait that value at index shift+(curwidx-1)*warpsz+lidx has been written (this avoids to relaunch a kernel)
            if flag[shift+(curwidx-1)*warpsz+lidx] == targetflag
                break
            end
            threadfence() # !!Must be in the loop otherwise the result is undefined
        end

        val = partial[shift+(curwidx-1)*warpsz+lidx]

        for u in powers_of_two(Val(warpsz))
            val_shuffled = shfl_up_sync(0xffffffff, val, u)

            if lidx > u
                val = op(val, val_shuffled)
            end
        end
        shift += curlength

        if lidx == warpsz || (curwidx - 1) * warpsz + lidx == curlength
            shifted_curwidx = shift + curwidx
            if shifted_curwidx == glmemlength
                result[1] = val
                return
            else
                partial[shifted_curwidx] = val
                threadfence()

                flag[shifted_curwidx] = targetflag
            end

        end

        (curwidx - 1) % warpsz != 0 && return


        curlength = cld(curlength, warpsz)
        curwidx = cld(curwidx, warpsz)

        #break
    end
end

struct MapReduceConfig{F<:Function,O<:Function}
    kernel::CUDA.HostKernel
    config::@NamedTuple{blocks::Int64, threads::Int64} #Should be the config given by the kernel
    shmemsize::Int
    glmemlength::Int
    f::F
    op::O
    lengthVs::Int
    T::DataType #We put the type here and not in MapReduce{T} so that it can dynamically change. This may create some overhead.
end

struct MapReduceGlmem{T}
    partial::AbstractGPUVector{T}
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

function (mpr::MapReduce)(f::F, op::O, result::AbstractGPUVector{T}, Vs::Tuple{Vararg{AbstractGPUVector{T}}}; reinit=false) where {T,F<:Function,O<:Function}
    reinit = reinit || mpr.config === nothing || T !== mpr.config.T || typeof(mpr.config.f) !== F || typeof(mpr.config.op) !== O || length(Vs) != mpr.config.lengthVs
    if reinit # Kernel config
        kernel = @cuda launch = false mapreducekernel(f, op, result, Vs, Vs[1], FLAG, FLAG_TYPE(0), 200)
        config = launch_configuration(kernel.fun; shmem=(threads) -> 32 * sizeof(T))
        mpr.config = MapReduceConfig(
            kernel,
            config,
            32 * sizeof(T),
            total_glmem_length(config.blocks),
            f, op,
            length(Vs),
            T,
        )
    end

    if reinit || (mpr.glmem === nothing) # Global memory allocation
        l = mpr.config.glmemlength
        mpr.glmem = MapReduceGlmem(
            CuArray{T}(undef, l),
            CuArray{FLAG_TYPE}(undef, l) # Allocating two cuArray is costly, (∼ 1.5μs). Can we do this in one go ? The pb is they have not same type
        )
    end

    N = length(Vs[1])
    threads = min(mpr.config.config.threads, N)
    blocks = min(mpr.config.config.blocks, cld(N, threads))
    targetflag = rand(FLAG_TYPE)

    (mpr.config.kernel)(
        mpr.config.f, mpr.config.op,
        result,
        Vs,
        mpr.glmem.partial,
        mpr.glmem.flag,
        targetflag,
        mpr.config.glmemlength;
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


