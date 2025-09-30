include("helpers.jl")
using BenchmarkTools
function mapreducekernel(
    f::F, op::O,
    result::AbstractVector{Outf},
    Vs::NTuple{K,AbstractVector{T}},
    partial::AbstractVector{Outf},
    flag::AbstractVector{FLAG_TYPE},
    targetflag::FLAG_TYPE
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

    lane = Int(laneid()) #lane
    wid = (tid - 1) ÷ warpsz + 1 #warp index in the block

    #nwp = cld(threads, warpsz)

    chunksize = cld(n, blocks)
    shmem_res = @cuDynamicSharedMem(Outf, 32)#Dynamic mem uses more registers

    block_start = (bid - 1) * chunksize + 1
    block_end = min(block_start + chunksize - 1, n)

    block_start > n && return

    @inbounds if length(Vs) == 1
        i = block_start + tid - 1
        val = f(Vs[1][min(i, block_end)])
        last = block_end - 4 * threads
        while i <= last
            i += threads
            v1 = f(Vs[1][i])
            i += threads
            v2 = f(Vs[1][i])
            i += threads
            v3 = f(Vs[1][i])
            i += threads
            v4 = f(Vs[1][i])

            val = op(val, v1)
            val = op(val, v2)
            val = op(val, v3)
            val = op(val, v4)
        end
        i += threads
        while i <= block_end
            val = op(val, f(Vs[1][i]))
            i += threads
        end
    elseif length(Vs) == 2
        i = block_start + tid - 1
        val = f(Vs[1][min(i, block_end)], Vs[2][min(i, block_end)])
        last = block_end - 2 * threads
        while i <= last
            i += threads
            v1 = f(Vs[1][i], Vs[2][i])#
            i += threads
            v2 = f(Vs[1][i], Vs[2][i])
            val = op(val, v1)
            val = op(val, v2)
        end
        i += threads
        while i <= block_end
            val = op(val, f(Vs[1][i]), Vs[2][i])
            i += threads
        end
    else
        val = f((V[min(block_start + tid - 1, block_end)] for V in Vs)...)# Somehow less optimized than two previous functions
        #i = block_start + threads + tid - 1
        for i in (block_start+threads+tid-1:threads:block_end)
            val = op(val, f((Vs[k][i] for k in (1:length(Vs)))...))#
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

    @inbounds if ((lane == warpsz && tid < block_end - block_start + 1) || (tid == block_end - block_start + 1))
        shmem_res[wid] = val
    end

    if threads <= block_end - block_start + 1
        wid != cld(threads, warpsz) && return
    else
        (wid != cld(block_end - block_start + 1, warpsz)) && return
    end

    sync_threads()

    @inbounds val = shmem_res[lane]

    offset = 0x00000001
    while offset < warpsz
        shuffled = shfl_up_sync(0xffffffff, val, offset)
        if lane > offset
            val = op(val, shuffled)
        end
        offset <<= 1
    end

    @inbounds if (
        (threads <= block_end - block_start + 1 && lane == cld(threads, warpsz)) ||
        (threads > block_end - block_start + 1 && (lane == cld(block_end - block_start + 1, warpsz)))
    )
        if blocks == 1
            result[1] = val
            return
        end
        partial[bid] = val
        threadfence()
        flag[bid] = targetflag
    end

    bid % warpsz != 1 && return #only keep first warp of each block


    clength = blocks # current number of warps
    cwid = cld(bid, warpsz) # current index of the warp

    shift = 0
    @inbounds while clength != 1
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

            if cld(clength, warpsz) == 1
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
Vs = (V, w)
f = *
opp = +

partial = CuArray{T}(undef, 0)
flag = CuArray{FLAG_TYPE,1,CUDA.DeviceMemory}(undef, 0)
#flag = CuArray{FLAG_TYPE,1,CUDA.DeviceMemory}(undef, glmemlength)

targetflag = rand(FLAG_TYPE)

kernel = @cuda launch = false mapreducekernel(f, opp, result, Vs, partial, flag, targetflag)
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
CUDA.@sync kernel(f, opp, result, Vs, partial, flag, targetflag; shmem=shmem, threads=threads, blocks=blocks)
result, sum(1:N)

#%%
x = CUDA.@profile CUDA.@sync kernel(f, opp, result, Vs, partial, flag, targetflag, glmemlength; shmem=shmem, threads=threads, blocks=blocks)
#%%
@btime CUDA.@sync kernel($f, $opp, $result, $Vs, $partial, $flag, $targetflag; shmem=shmem, threads=threads, blocks=blocks)
#%%
CUDA.@time kernel(f, opp, result, Vs, partial, flag, targetflag; shmem=shmem, threads=threads, blocks=blocks)

x.device.dt = (x.device.stop - x.device.start) * 1e6
CSV.write("./test.csv", x.device)
x.device.registers
#x = CUDA.@elapsed CUDA.@sync kernel(f, opp, result, Vs, partial, flag, targetflag, glmemlength; shmem=shmem, threads=threads, blocks=blocks)
x

#%%
x = CUDA.@bprofile time = 0.05 CUDA.@sync CUDA.dot(V, w)
x = CUDA.@elapsed CUDA.@sync CUDA.dot(V, w)
u = CUDA.@time CUDA.@sync CUDA.dot(V, w)
CUDA.@timed CUDA.@sync CUDA.dot(V, w)
x.device.dt = (x.device.stop - x.device.start) * 1e6
#%%
CSV.write("./test_cublas.csv", x.device)
#%%
using CSV
CSV.write("./test.csv", x.device)
#%%
t = 0
function bench(f, opp, result, Vs, partial, flag, targetflag; shmem=shmem, threads=threads, blocks=blocks)
    t = 0
    for i in (1:10000)
        u = time()
        CUDA.@sync kernel(f, opp, result, Vs, partial, flag, targetflag; shmem=shmem, threads=threads, blocks=blocks)
        t += time() - u
    end
    return t / 10000
end
bench(f, opp, result, Vs, partial, flag, targetflag) * 1e6
@btime