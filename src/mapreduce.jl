include("../helpers.jl")



using CUDA, Random
import CUDA: AbstractGPUVector

const FLAG_TYPE = UInt128 # Risk of error of order 1/2^64
const FLAG = CuArray{FLAG_TYPE,1,CUDA.DeviceMemory}(undef, 0) # This is to determine the parameters of the kernel without redefining a FLAG CuArray each time.


function mapreducekernel(
    f::F, op::O,
    result::AbstractVector{T},
    Vs::NTuple{K,AbstractVector{T}},
    partial::AbstractVector{T},
    flag::AbstractVector{FLAG_TYPE},
    targetflag::FLAG_TYPE,
    glmemlength::Int,
    ::Outf
) where {F<:Function,O<:Function,K,T,Outf}
    n = UInt(length(Vs[1]))
    blocks = UInt32(gridDim().x)
    threads = UInt32(blockDim().x)
    warpsz = 0x0020
    bid = UInt32(blockIdx().x)
    tid = UInt32(threadIdx().x)

    strd = threads * blocks
    gid = (bid - 0x01) * threads + tid

    # It is tempting to write (gid > n && return). This "breaks" the last warp however and we cannot synchronize threads if we do that...

    lane = ((tid - 0x01) % warpsz + 0x01) #lane
    wid = (tid - 0x01) ÷ warpsz + 0x01 #warp index in the block

    nwp = cld(threads, warpsz)


    shmem_res = @cuDynamicSharedMem(Outf, 32)
    m = min(gid, n)

    @inbounds if length(Vs) == 1
        val = f(Vs[1][m])
        for i in gid+strd:strd:n
            val = op(val, f(Vs[1][i]))#
        end
    elseif length(Vs) == 2
        val = f(Vs[1][m], Vs[2][m])
        for i in gid+strd:strd:n
            val = op(val, f(Vs[1][i], Vs[2][i]))#
        end
    else
        val = op(val, f((V[m] for V in Vs)...))# Somehow less optimized than two previous functions
        for i in gid+strd:strd:n
            val = op(val, f((V[i] for V in Vs)...))#
        end
    end
    #return val
    #val = strided_sum(f, op, Vs, gid, strd, n, Val(K))

    for u in powers_of_two(Val(warpsz))
        shuffled = shfl_up_sync(0xffffffff, val, u)
        if lane > u
            val = op(val, shuffled)
        end
    end

    @inbounds if ((lane == warpsz && gid < n) || (gid == n))
        shmem_res[wid] = val
    end

    if strd <= n
        wid != nwp && return
    else
        maxnwp = cld(n, warpsz) # max number of useful warps
        gwid = (gid - 0x01) ÷ warpsz + 0x01 # global warp index
        gwid > maxnwp && return
        (wid != nwp && gwid != maxnwp) && return
    end

    sync_threads()

    @inbounds val = shmem_res[lane]

    for u in powers_of_two(Val(warpsz))
        shuffled = shfl_up_sync(0xffffffff, val, u)
        if lane > u
            val = op(val, shuffled)
        end
    end
    filtr = (
        (strd <= n && lane == nwp) ||
        (strd > n && (lane == nwp && bid < blocks || lane == cld((n - 0x01) % threads + 0x01, warpsz) && bid == blocks))
    )
    @inbounds if filtr
        partial[bid] = val
        threadfence()
        flag[bid] = targetflag
    end

    bid % warpsz != 1 && return

    @inbounds if blocks == 1
        if filtr
            result[1] = val
        end
        return
    end

    clength = blocks # current number of warps
    cwid = cld(bid, warpsz) # current index of the warp

    shift = 0x00
    @inbounds while clength != 1
        cid = (cwid - 0x01) * warpsz + lane
        while true #Wait that value at index shift+cid has been written (this avoids to relaunch a kernel)
            if cid > clength || flag[shift+cid] == targetflag
                break
            end
            threadfence() # !!Must be in the loop otherwise the result is undefined
        end
        val = partial[shift+cid]
        for u in powers_of_two(Val(warpsz))
            val_shuffled = shfl_up_sync(0xffffffff, val, u)
            if lane > u
                val = op(val, val_shuffled)
            end
        end
        shift += clength

        if cid <= clength && ((lane == warpsz) || (cid == clength))
            shifted_cwid = shift + cwid

            if shifted_cwid == glmemlength
                result[1] = val
                return
            else
                partial[shifted_cwid] = val
                threadfence()
                flag[shifted_cwid] = targetflag
            end

        end

        cwid % warpsz != 1 && return

        clength = cld(clength, warpsz)
        cwid = cld(cwid, warpsz)

    end
    return
end

struct MapReduceConfig{F<:Function,O<:Function,T,Outf}
    kernel::CUDA.HostKernel
    config::@NamedTuple{blocks::Int, threads::Int} #Should be the config given by the kernel
    shmemsize::Int
    f::F
    op::O
    lengthVs::Int
    source::T
    out::Outf # Example of out type of f: T -> Outf.
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

function (mpr::MapReduce)(f::F, op::O, result::AbstractGPUVector, Vs::NTuple{K,AbstractGPUVector{T}}; reinit=false) where {T,K,F<:Function,O<:Function}
    reinit = reinit || mpr.config === nothing || T !== typeof(mpr.config.source) || typeof(mpr.config.f) !== F || typeof(mpr.config.op) !== O || K != mpr.config.lengthVs
    if reinit # Kernel config
        bytes = zeros(UInt8, sizeof(T))
        source = reinterpret(T, bytes)[1]
        out = f((source for _ in (1:K))...) # Example of an element of right type
        kernel = @cuda launch = false mapreducekernel(f, op, result, Vs, Vs[1], FLAG, FLAG_TYPE(0), 200, out)
        config = launch_configuration(kernel.fun; shmem=(threads) -> 32 * sizeof(T))
        mpr.config = MapReduceConfig(
            kernel,
            config,
            32 * sizeof(out),
            f, op,
            K,
            source,
            out
        )
    end

    if reinit || (mpr.glmem === nothing) # Global memory allocation
        glmemlength = total_glmem_length(Val(mpr.config.config.blocks), Val(32)) # ***the actual number of blocks glmemlength used in kernel can be smaller
        mpr.glmem = MapReduceGlmem(
            CuArray{T}(undef, glmemlength),
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
        glmemlength,
        mpr.config.out;
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


