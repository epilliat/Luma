


using CUDA, Random
import CUDA: AbstractGPUVector


FLAG_TYPE = UInt128 # Risk of error of order 1/2^64
PTR_TYPE = Int
FENCE_TYPE = UInt16

FLAG_AR = CuArray{FLAG_TYPE,1,CUDA.DeviceMemory}(undef, 0) # This is to determine the parameters of the kernel without redefining a FLAG CuArray each time.
PTR_AR = CuArray{PTR_TYPE,1,CUDA.DeviceMemory}(undef, 0) # This is to determine the parameters of the kernel without redefining a FLAG CuArray each time.
FENCE_MAT = CuMatrix{PTR_TYPE,CUDA.DeviceMemory}(undef, (0, 0)) # This is to determine the parameters of the kernel without redefining a FLAG CuArray each time.



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