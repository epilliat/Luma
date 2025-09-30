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

    strd = threads * blocks
    gid = (bid - 1) * threads + tid

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
        val = f(Vs[1][block_start+tid-1])
        i = block_start + threads + tid - 1
        while i <= block_end
            val = op(val, f(Vs[1][i]))#
            i += threads
        end
    elseif length(Vs) == 2
        val = f(Vs[1][block_start+tid-1], Vs[2][block_start+tid-1])
        i = block_start + threads + tid - 1
        while i <= block_end
            val = op(val, f(Vs[1][i], Vs[2][i]))#
            i += threads
        end
    else
        val = f((V[block_start+tid-1] for V in Vs)...)# Somehow less optimized than two previous functions
        #i = block_start + threads + tid - 1
        for i in (block_start+threads+tid-1:threads:block_end)
            val = f((Vs[k][i] for k in (1:length(Vs)))...)#
            i += threads
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

    @inbounds if ((lane == warpsz && tid < block_end - block_start + 1) || (tid == block_end - block_start + 1))
        shmem_res[wid] = val
    end


    jump_to_block_reduction = (threads <= block_end - block_start + 1 && wid != cld(threads, warpsz)
                               ||
                               threads > block_end - block_start + 1 && wid != cld(block_end - block_start + 1, warpsz))
    sync_threads()
    if !jump_to_block_reduction

        @inbounds val = shmem_res[lane]

        for u in powers_of_two(Val(warpsz))
            shuffled = shfl_up_sync(0xffffffff, val, u)
            if lane > u
                val = op(val, shuffled)
            end
        end
        filtr = (
            (threads <= block_end - block_start + 1 && lane == cld(threads, warpsz)) ||
            (threads > block_end - block_start + 1 && (lane == cld(block_end - block_start + 1, warpsz)))
        )
        @inbounds if filtr
            partial[bid] = val
            threadfence()
            flag[bid] = targetflag
        end
    end

    ###===== BLOCK REDUCTION =====###

    clength = blocks
    shift = 0

    @inbounds while true

        sync_threads()
        wid > cld(clength, warpsz) && return

        while true #Wait that value at index shift+cid has been written (this avoids to relaunch a kernel)
            if gid > clength || flag[shift+gid] == targetflag
                break
            end
            threadfence() # !!Must be in the loop otherwise the result is undefined
        end
        val = partial[min(shift + gid, shift + clength)]
        for u in powers_of_two(Val(warpsz))
            val_shuffled = shfl_up_sync(0xffffffff, val, u)
            if lane > u
                val = op(val, val_shuffled)
            end
        end


        if ((lane == warpsz && gid < clength) || (gid == clength))
            shmem_res[wid] = val
        end

        sync_threads()

        val = shmem_res[lane]

        for u in powers_of_two(Val(warpsz))
            shuffled = shfl_up_sync(0xffffffff, val, u)
            if lane > u
                val = op(val, shuffled)
            end
        end

        shift += clength
        if clength == 1
            break
        end


        if !(threads < clength && wid != cld(threads, warpsz)
             ||
             threads >= clength && wid != cld(clength, warpsz))
            if (threads < clength && lane == cld(threads, warpsz)
                ||
                threads >= clength && lane == cld(clength, warpsz))

                partial[shift+bid] = val
                threadfence()
                flag[shift+bid] = targetflag
            end
        end
        clength = cld(clength, threads)


    end
    if gid == 1
        result[1] = val
    end
    return
end

N = 1000000

#cld(threads, 32)
T = Float64
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
#threads = 128
#blocks = 40
glmemlength = total_glmem_length(Val(blocks), Val(threads))
partial = CuArray{T}(undef, glmemlength)

flag = CuArray{FLAG_TYPE,1,CUDA.DeviceMemory}(undef, glmemlength)
shmem = 32 * sizeof(T)


#CUDA.@sync @cuda shmem = shmem_size threads = threads blocks = blocks mapreducekernel(f, opp, result, Vs, partial, flag, targetflag, glmemlength, 0.0)
println("computing kernel")

CUDA.@sync kernel(f, opp, result, Vs, partial, flag, targetflag, glmemlength; shmem=shmem, threads=threads, blocks=blocks)
sum(partial), sum(1:N), result

#%%
@btime CUDA.@sync kernel($f, $opp, $result, $Vs, $partial, $flag, $targetflag, $glmemlength; shmem=shmem, threads=threads, blocks=blocks)
CUDA.@profile CUDA.@sync kernel(f, opp, result, Vs, partial, flag, targetflag, glmemlength; shmem=shmem, threads=threads, blocks=blocks)
