function mapreducekernel(
    f::F, op::O,
    result::AbstractVector{Outf},
    Vs::NTuple{K,AbstractVector{T}},
    partial::AbstractVector{Outf},
    flag::AbstractVector{FLAG_TYPE},
    targetflag::FLAG_TYPE,
    Left::Integer, Right::Integer, acc::Bool
) where {F<:Function,O<:Function,K,T,Outf}
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

    chunksize = cld(Right - Left + 1, blocks)
    shmem_res = @cuDynamicSharedMem(Outf, 32)#Dynamic mem uses more registers

    block_start = (bid - 1) * chunksize + Left
    block_end = min(block_start + chunksize - 1, Right)

    block_start > Right && return

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
            if acc
                result[1] = op(result[1], val)
            else
                result[1] = val
            end
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
                if acc
                    result[1] = op(result[1], val)
                else
                    result[1] = val
                end
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
