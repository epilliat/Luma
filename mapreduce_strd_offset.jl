include("helpers.jl")
using BenchmarkTools
function mapreducekernel(
    f::F, op::O,
    result::AbstractVector{Outf},
    Vs::NTuple{K,AbstractVector{T}},
    partial::AbstractVector{Outf},
    flag::AbstractVector{FLAG_TYPE},
    targetflag::FLAG_TYPE,
    glmemlength::Int,
) where {F<:Function,O<:Function,K,T,Outf}
    n = Int(length(Vs[1]))
    blocks = Int(gridDim().x)
    threads = Int(blockDim().x)
    warpsz = 32
    bid = Int(blockIdx().x)
    tid = Int(threadIdx().x)

    #strd = threads * blocks
    #gid = (bid - 1) * threads + tid

    # It is tempting to write (gid > n && return). This "breaks" the last warp however and we cannot synchronize threads if we do that...

    lane = ((tid - 1) % warpsz + 1) #lane
    wid = (tid - 1) ÷ warpsz + 1 #warp index in the block

    #nwp = cld(threads, warpsz)

    chunksize = cld(n, blocks)
    shmem_res = @cuDynamicSharedMem(Outf, 32)

    block_start = (bid - 1) * chunksize + 1
    block_end = min(block_start + chunksize - 1, n)

    block_start > n && return

    @inbounds if length(Vs) == 1
        val = f(Vs[1][min(block_start + tid - 1, block_end)])
        i = block_start + threads + tid - 1
        while i <= block_end
            val = op(val, f(Vs[1][i]))#
            i += threads
        end
    elseif length(Vs) == 2
        val = f(Vs[1][min(block_start + tid - 1, block_end)], Vs[2][min(block_start + tid - 1, block_end)])
        i = block_start + threads + tid - 1
        while i <= block_end
            val = op(val, f(Vs[1][i], Vs[2][i]))#
            i += threads
        end
    else
        val = f((V[min(block_start + tid - 1, block_end)] for V in Vs)...)# Somehow less optimized than two previous functions
        #i = block_start + threads + tid - 1
        for i in (block_start+threads+tid-1:threads:block_end)
            val = f((Vs[k][i] for k in (1:length(Vs)))...)#
            i += threads
        end
    end
    #return val
    #val = strided_sum(f, op, Vs, gid, strd, n, Val(K))
    offset = 0x00000001
    while offset < warpsz
        shuffled = shfl_up_sync(0xffffffff, val, offset)
        if lane > offset
            val = op(val, shuffled)
        end
        offset <<= 1
    end

    if ((lane == warpsz && tid < block_end - block_start + 1) || (tid == block_end - block_start + 1))
        shmem_res[wid] = val
    end

    if threads <= block_end - block_start + 1
        wid != cld(threads, warpsz) && return
    else
        (wid != cld(block_end - block_start + 1, warpsz)) && return
    end

    sync_threads()

    val = shmem_res[lane]

    offset = 0x00000001
    while offset < warpsz
        shuffled = shfl_up_sync(0xffffffff, val, offset)
        if lane > offset
            val = op(val, shuffled)
        end
        offset <<= 1
    end
    filtr = (
        (threads <= block_end - block_start + 1 && lane == cld(threads, warpsz)) ||
        (threads > block_end - block_start + 1 && (lane == cld(block_end - block_start + 1, warpsz)))
    )
    if filtr
        partial[bid] = val
        threadfence()
        flag[bid] = targetflag
    end

    bid % warpsz != 1 && return #only keep first warp of each block

    if blocks == 1
        if filtr
            result[1] = val
        end
        return
    end

    clength = blocks # current number of warps
    cwid = cld(bid, warpsz) # current index of the warp

    shift = 0
    while clength != 1
        cid = (cwid - 1) * warpsz + lane
        while true #Wait that value at index shift+cid has been written (this avoids to relaunch a kernel)
            if cid > clength || flag[shift+cid] == targetflag
                break
            end
            threadfence() # !!Must be in the loop otherwise the result is undefined
        end
        val = partial[min(shift + cid, shift + clength)]
        offset = 0x00000001
        while offset < warpsz
            shuffled = shfl_up_sync(0xffffffff, val, offset)
            if lane > offset
                val = op(val, shuffled)
            end
            offset <<= 1
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

N = 1000000

#cld(threads, 32)
T = Float32
V = CuArray{T}(1:N)
#V = CUDA.rand(N)
result = CuArray{T}([0.0])
#test = CUDA.fill(0.0, 40000)
w = CUDA.ones(T, N)
Vs = (V,)
f = *
opp = +

partial = CuArray{T}(undef, 0)
flag = CuArray{FLAG_TYPE,1,CUDA.DeviceMemory}(undef, 0)
#flag = CuArray{FLAG_TYPE,1,CUDA.DeviceMemory}(undef, glmemlength)

targetflag = rand(FLAG_TYPE)

kernel = @cuda launch = false mapreducekernel(f, opp, result, Vs, partial, flag, targetflag, 200)
config = launch_configuration(kernel.fun)
threads = min(config.threads, N)
threads = config.threads
#threads = 32


blocks = min(config.blocks, cld(N, threads))
#blocks = min(10, cld(N, threads))
#blocks = 20
strd = blocks * threads
#threads=320
#blocks=40
glmemlength = total_glmem_length(Val(blocks), Val(32))
partial = CuArray{T}(undef, glmemlength)

flag = CuArray{FLAG_TYPE,1,CUDA.DeviceMemory}(undef, glmemlength)
shmem = 32 * sizeof(T)

#CUDA.@sync @cuda shmem = shmem_size threads = threads blocks = blocks mapreducekernel(f, opp, result, Vs, partial, flag, targetflag, glmemlength, 0.0)
CUDA.@sync kernel(f, opp, result, Vs, partial, flag, targetflag, glmemlength; shmem=shmem, threads=threads, blocks=blocks)
result, sum(1:N)

#%%
@btime CUDA.@sync kernel($f, $opp, $result, $Vs, $partial, $flag, $targetflag, $glmemlength; shmem=shmem, threads=threads, blocks=blocks)
CUDA.@profile CUDA.@sync kernel(f, opp, result, Vs, partial, flag, targetflag, glmemlength; shmem=shmem, threads=threads, blocks=blocks)
