@generated function powers_of_two(::Val{N})::Tuple{Vararg{UInt32}} where {N}
    # Input validation
    if N <= 0
        throw(ArgumentError("Input N must be positive"))
    end

    # Calculate ceil(log2(N)) using the simpler function
    log2_ceiling = ceil(Int, log2(N >> 1))

    V = [UInt32(2^i) for i in 0:log2_ceiling]

    # Create expression to return a tuple of the V
    return :($(Expr(:tuple, V...)))
end

function total_glmem_length(blocks; warpsize=32)
    total = remainder = blocks
    while remainder != 1
        remainder = cld(remainder, warpsize)
        total += remainder
    end
    return total
end