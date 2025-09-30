function accumulatekernel(
    f, op,
    result::AbstractVector{Outf},
    V::AbstractVector{T},
    partial::AbstractVector{Outf},
    flag::AbstractVector{<:Integer},
    targetflag::Integer,
    ptrs::AbstractVector{<:Integer},
    fences::AbstractMatrix{<:Integer},
) where {T,Outf}
    n = length(V)
    blocks = Int(gridDim().x)
    threads = Int(blockDim().x)

    warpsz = 32
    bid = Int(blockIdx().x)
    tid = Int(threadIdx().x)
    strd = threads * blocks
    gid = (bid - 1) * threads + tid

    lane = ((tid - 1) % warpsz + 1) # lane
    wid = (tid - lane) ÷ warpsz + 1 # warp index in the block
    gwid = (gid - 1) ÷ warpsz + 1 # warp index in the grid

    shmem_res = @cuDynamicSharedMem(Outf, 32)

    k = cld(n, strd) * warpsz # size of an interval block [l,r] on which a warp computes the accumulation
    l = k * (gwid - 1) + 1
    r = k * gwid

    L = (bid - 1) * threads * cld(n, strd) + 1 # l of first warp of the block

    L > n && return
    m = min(r, n)
    filtr = l <= m

    startchunk = l
    #val = V[startchunk+lane-1]
    for chunk in (1:cld((m - l + 1), warpsz))
        chlane = startchunk + lane - 1
        val = f(V[min(chlane, m)])
        for u in powers_of_two(Val(warpsz))
            shuffled = shfl_up_sync(0xffffffff, val, u)
            if lane > u
                val = op(val, shuffled)
            end
        end
        if chlane <= m
            if chunk == 1
                result[chlane] = val
            else #&& chlane <= r
                result[chlane] = op(result[startchunk-1], val)
            end
        end
        startchunk += warpsz
    end
    val = result[m]
    last_warp = min(cld(n - L + 1, k), cld(threads, warpsz))
    if (lane == (m - l) % warpsz + 1) #&& wid <= last_warp
        shmem_res[wid] = val
    end

    filtr &= (wid == last_warp)

    sync_threads()###
    val = shmem_res[lane]
    for u in powers_of_two(Val(warpsz))
        shuffled = shfl_up_sync(0xffffffff, val, u)
        if lane > u
            val = op(val, shuffled)
        end
    end
    shmem_res[lane] = val ## accumulation of warp results in shmem_res ! We will need that when we reverse the tree

    if filtr && lane == last_warp
        partial[bid] = val
        #test[bid] = val
    end

    @inbounds if wid == 1
        ptrs[bid] = bid

        fence = 0x00
        fences[bid, 1] = fence
        fences[bid, 2] = fence

        ptleft = ptrs[bid]

        threadfence()
        flag[bid] = targetflag

        while ptrs[bid] > 1
            #cpt += 1
            fence += 0x01

            ptleft = ptrs[bid] - 1

            #threadfence()
            flag_ptleft = flag[ptleft]
            (flag_ptleft == targetflag || flag_ptleft == targetflag + 0x01) || continue
            #threadfence()

            fence_ptleft2 = fences[ptleft, 2]
            threadfence()

            pt_ptleft = ptrs[ptleft]
            partial_ptleft = partial[ptleft]

            threadfence()
            fence_ptleft1 = fences[ptleft, 1]


            if fence_ptleft1 == fence_ptleft2 && lane == 1 # We put condition on lane here for CUDA optimization purpose
                fences[bid, 1] = fence
                threadfence()

                ptrs[bid] = pt_ptleft
                partial[bid] = op(partial[bid], partial_ptleft)

                threadfence()
                fences[bid, 2] = fence

            end
        end

        #threadfence()
        flag[bid] = targetflag + 0x01
    end


    if bid == 1
        if wid > 1
            @inbounds offset = shmem_res[wid-1]
        end
    else
        @inbounds while true
            if flag[bid-1] == targetflag + 0x01
                break
            end
            threadfence()
        end
        offset = partial[bid-1]
        if wid != 1
            offset = op(offset, shmem_res[wid-1])
        end
    end
    if wid > 1 || bid > 1
        @inbounds for i in (l+lane-1:warpsz:min(r, n))
            result[i] = op(result[i], offset)
        end
    end

    return
end