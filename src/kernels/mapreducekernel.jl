function mapreducekernel(
    f::F, op::O,
    result::AbstractVector{Outf},
    Vs::NTuple{K,AbstractVector{T}},
    partial::AbstractVector{Outf},
    flag::AbstractVector{FLAG_TYPE},
    targetflag::FLAG_TYPE,
    glmemlength::Int,
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