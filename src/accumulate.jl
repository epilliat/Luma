include("../helpers.jl")
include("kernels/accumulatekernel.jl")



struct AccumulateConfig{F<:Function,O<:Function,T}
    kernel::CUDA.HostKernel
    config::@NamedTuple{blocks::Int, threads::Int} #Should be the config given by the kernel
    shmemsize::Int
    f::F
    op::O
    source::T
end

struct AccumulateGlmem{Outf}
    partial::AbstractGPUVector{Outf}
    flag::AbstractGPUVector{FLAG_TYPE}
    ptrs::AbstractGPUVector{<:Integer}
    fences::AbstractGPUMatrix{<:Integer} # first and second fences
end

mutable struct Accumulate
    config::Union{AccumulateConfig,Nothing}
    glmem::Union{AccumulateGlmem,Nothing}
    storeGlmem::Bool
end

Accumulate(; storeGlmem=true) = Accumulate(nothing, nothing, storeGlmem)


function (acc::Accumulate)(f::F, op::O, result::AbstractGPUVector{Outf}, V::AbstractGPUVector{T}; reinit=false) where {T,Outf,F<:Function,O<:Function}
    reinit = reinit || acc.config === nothing || T !== typeof(acc.config.source) || typeof(acc.config.f) !== F || typeof(acc.config.op) !== O
    if reinit # Kernel config
        bytes = zeros(UInt8, sizeof(T))
        source = reinterpret(T, bytes)[1]

        kernel = @cuda launch = false accumulatekernel(f, op, result, V, result, FLAG_AR1, FLAG_TYPE(0), PTRS_AR1, FENCES_AR2)
        config = launch_configuration(kernel.fun; shmem=(threads) -> 32 * sizeof(Outf))
        acc.config = AccumulateConfig(
            kernel,
            config,
            32 * sizeof(Outf),
            f, op,
            source
        )
    end

    if reinit || (acc.glmem === nothing) # Global memory allocation
        blocks = acc.config.config.blocks # blocks is larger than effective blocks used in kernel (if n is small)
        acc.glmem = AccumulateGlmem(
            CuArray{Outf}(undef, blocks),
            CuArray{FLAG_TYPE}(undef, blocks),
            CuArray{PTR_TYPE}(undef, blocks),
            CuArray{FENCE_TYPE}(undef, (blocks, 2))
        )
    end

    N = length(V)
    threads = min(acc.config.config.threads, N)
    blocks = min(acc.config.config.blocks, cld(N, threads))
    targetflag = rand(FLAG_TYPE)
    (acc.config.kernel)(
        acc.config.f, acc.config.op,
        result,
        V,
        acc.glmem.partial,
        acc.glmem.flag,
        targetflag,
        acc.glmem.ptrs,
        acc.glmem.fences;
        shmem=acc.config.shmemsize,
        threads=threads,
        blocks=blocks
    )

    if !acc.storeGlmem
        CUDA.unsafe_free!(acc.glmem.partial)
        CUDA.unsafe_free!(acc.glmem.flag)
        CUDA.unsafe_free!(acc.glmem.ptrs)
        CUDA.unsafe_free!(acc.glmem.fences)
        acc.glmem = nothing
    end
end