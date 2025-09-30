include("./helper.jl")
include("../src/mapreduce.jl")
using CUDA
using BenchmarkTools
using KernelAbstractions
import AcceleratedKernels as AK
tmax_timed = 1
names = ["Cublas", "Luma", "CUDA.jl", "AK"]
algos = ["Sum", "Dot"]
bench = DataFrame()

#%%
#====================== DOT Product===================#

algo = "Dot"


T = Float32
N = Int(1e8)
V = CuArray{T}(1:N |> collect)
w = CUDA.ones(T, N)
Vs = (V, w)
result = CuArray{T}([0.0]) # We store the result in the GPU
result_unified = CuArray{T,1,CUDA.UnifiedMemory}([0.0]) # Or in the CPU (less efficient)
#%%
name = "Cublas"
start_time = time()
while time() - start_time < 0.500  # 500ms warm-up
    CUDA.@sync CUDA.dot(V, w)
end
prof = CUDA.@bprofile time = 0.02 CUDA.dot(V, w)
dt = 0
dts = []
while dt <= 2 * tmax_timed
    timed = (CUDA.@timed CUDA.dot(V, w))#(CUDA.@timed CUDA.@sync eval(exp_expr))[:time]
    u = timed[:time]
    dt += u
    push!(dts, u)
end
timed = (CUDA.@timed CUDA.@sync CUDA.dot(V, w))
benchmark_summary!(prof, timed, dts, T, N, name, "Sum", bench)
benchmark_summary!(prof, timed, dts, T, N, name, "Dot", bench)
#push all the infos
#%%
name = "Luma"
mpr = MapReduce(storeGlmem=true)
mpr(*, +, result, Vs) # Kernel and Global memory are initialized at first run 
start_time = time()
while time() - start_time < 0.500  # 500ms warm-up
    CUDA.@sync mpr(*, +, result, Vs)
end
prof = CUDA.@bprofile time = 0.05 mpr(*, +, result, Vs)
dt = 0
dts = []
while dt <= 2 * tmax_timed
    timed = (CUDA.@timed mpr(*, +, result, Vs))#(CUDA.@timed CUDA.@sync eval(exp_expr))[:time]
    u = timed[:time]
    dt += u
    push!(dts, u)
end
timed = (CUDA.@timed CUDA.@sync mpr(*, +, result, Vs))
benchmark_summary!(prof, timed, dts, T, N, name, algo, bench)

#%%
name = "CUDA.jl"
start_time = time()
while time() - start_time < 0.500  # 500ms warm-up
    CUDA.@sync mapreduce(identity, +, Vs[1] .* Vs[2])
end
prof = CUDA.@bprofile time = 0.05 mapreduce(identity, +, Vs[1] .* Vs[2])
dt = 0
dts = []
while dt <= 2 * tmax_timed
    timed = (CUDA.@timed mapreduce(identity, +, Vs[1] .* Vs[2]))#(CUDA.@timed CUDA.@sync eval(exp_expr))[:time]
    u = timed[:time]
    dt += u
    push!(dts, u)
end
timed = (CUDA.@timed CUDA.@sync mapreduce(identity, +, Vs[1] .* Vs[2]))
benchmark_summary!(prof, timed, dts, T, N, name, algo, bench)
@show bench.datatype


#%%
name = "AK"
start_time = time()
while time() - start_time < 0.500  # 500ms warm-up
    CUDA.@sync AK.mapreduce(identity, +, Vs[1] .* Vs[2]; init=T(0))
end
prof = CUDA.@bprofile time = 0.05 AK.mapreduce(identity, +, Vs[1] .* Vs[2]; init=T(0))
dt = 0
dts = []
while dt <= 2 * tmax_timed
    timed = (CUDA.@timed AK.mapreduce(identity, +, Vs[1] .* Vs[2]; init=T(0)))#(CUDA.@timed CUDA.@sync eval(exp_expr))[:time]
    u = timed[:time]
    dt += u
    push!(dts, u)
end
timed = (CUDA.@timed CUDA.@sync AK.mapreduce(identity, +, Vs[1] .* Vs[2]; init=T(0)))
benchmark_summary!(prof, timed, dts, T, N, name, algo, bench)
@show bench.datatype
#%%
#====================== Sum ===================#

algo = "L2 norm"

L2 = x -> x * x
T = Float64
N = Int(1e8)
V = CuArray{T}(1:N |> collect)
w = CUDA.ones(T, N)
result = CuArray{T}([0.0]) # We store the result in the GPU
result_unified = CuArray{T,1,CUDA.UnifiedMemory}([0.0]) # Or in the CPU (less efficient)

#push all the infos
#%%
name = "Luma"
mpr = MapReduce(storeGlmem=true)
mpr(L2, +, result, (V,)) # Kernel and Global memory are initialized at first run 
start_time = time()
while time() - start_time < 0.500  # 500ms warm-up
    CUDA.@sync mpr(L2, +, result, (V,))
end
prof = CUDA.@bprofile time = 0.05 mpr(L2, +, result, (V,))
dt = 0
dts = []
while dt <= 2 * tmax_timed
    timed = (CUDA.@timed mpr(L2, +, result, (V,)))#(CUDA.@timed CUDA.@sync eval(exp_expr))[:time]
    u = timed[:time]
    dt += u
    push!(dts, u)
end
timed = (CUDA.@timed CUDA.@sync mpr(L2, +, result, (V,)))
benchmark_summary!(prof, timed, dts, T, N, name, algo, bench)

#%%
name = "CUDA.jl"
start_time = time()
while time() - start_time < 0.500  # 500ms warm-up
    CUDA.@sync mapreduce(L2, +, V)
end
prof = CUDA.@bprofile time = 0.05 mapreduce(L2, +, V)
dt = 0
dts = []
while dt <= 2 * tmax_timed
    timed = (CUDA.@timed mapreduce(L2, +, V))#(CUDA.@timed CUDA.@sync eval(exp_expr))[:time]
    u = timed[:time]
    dt += u
    push!(dts, u)
end
timed = (CUDA.@timed CUDA.@sync mapreduce(L2, +, V))
benchmark_summary!(prof, timed, dts, T, N, name, algo, bench)
@show bench.datatype


#%%
name = "AK"
start_time = time()
while time() - start_time < 0.500  # 500ms warm-up
    CUDA.@sync AK.mapreduce(L2, +, V; init=T(0))
end
prof = CUDA.@bprofile time = 0.05 AK.mapreduce(L2, +, V; init=T(0))
dt = 0
dts = []
while dt <= 2 * tmax_timed
    timed = (CUDA.@timed AK.mapreduce(L2, +, V; init=T(0)))#(CUDA.@timed CUDA.@sync eval(exp_expr))[:time]
    u = timed[:time]
    dt += u
    push!(dts, u)
end
timed = (CUDA.@timed CUDA.@sync AK.mapreduce(L2, +, V; init=T(0)))
benchmark_summary!(prof, timed, dts, T, N, name, algo, bench)
@show bench.datatype

#%%
name = "Cublas"
start_time = time()
while time() - start_time < 0.500  # 500ms warm-up
    CUDA.@sync CUDA.norm(V)
end
prof = CUDA.@bprofile time = 0.02 CUDA.norm(V)
dt = 0
dts = []
while dt <= 2 * tmax_timed
    timed = (CUDA.@timed CUDA.norm(V))#(CUDA.@timed CUDA.@sync eval(exp_expr))[:time]
    u = timed[:time]
    dt += u
    push!(dts, u)
end
timed = (CUDA.@timed CUDA.@sync CUDA.norm(V))
benchmark_summary!(prof, timed, dts, T, N, name, algo, bench)