@generated function total_glmem_length(::Val{blocks}, ::Val{warpsize})::Int where {blocks,warpsize}
    total = rem = blocks
    while rem != 1
        rem = cld(rem, warpsize)
        total += rem
    end
    return :($total)
end

@generated function powers_of_two(::Val{N})::Tuple{Vararg{Int}} where {N}
    log2_ceiling = ceil(Int, log2(N >> 1))

    V = [2^i for i in 0:log2_ceiling]

    # Create expression to return a tuple of the V
    return :($(Expr(:tuple, V...)))
end