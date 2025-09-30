include("./helper.jl")
include("../src/mapreduce.jl")
using CUDA
using BenchmarkTools
tmax_timed = 1
names = ["Cublas", "Luma", "CUDA.jl", "KernelAbstractions"]
algos = ["Sum", "Dot"]
bench = Dict()


#====================== DOT Product===================#

algo = "Dot"


T = Float64
bench[T] = Dict(algo => Dict(name => Dict() for name in names) for algo in algos)
N = Int(1e6)
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
prof = CUDA.@bprofile time = 0.05 CUDA.dot(V, w)
dt = 0
dts = []
while dt <= 2 * tmax_timed
    timed = (CUDA.@timed CUDA.dot(V, w))#(CUDA.@timed CUDA.@sync eval(exp_expr))[:time]
    u = timed[:time]
    dt += u
    push!(dts, u)
end
timed = (CUDA.@timed CUDA.@sync CUDA.dot(V, w))
bench[T][algo][name][N] = benchmark_summary(prof, timed)
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
bench[T][algo][name][N] = benchmark_summary(prof, timed)


#%%
#====================== Sum ===================#

algo = "Dot"


T = Float64
bench[T] = Dict(algo => Dict(name => Dict() for name in names) for algo in algos)
N = Int(1e6)
V = CuArray{T}(1:N |> collect)
w = CUDA.ones(T, N)
Vs = (V, w)
result = CuArray{T}([0.0]) # We store the result in the GPU
result_unified = CuArray{T,1,CUDA.UnifiedMemory}([0.0]) # Or in the CPU (less efficient)
#%%
quantile(dts, 0.25)
eval(Meta.parse("CUDA.@timed " * exp_str))
T = Float64
N = Int(1e6)
V = CuArray{T}(1:N |> collect)
w = CUDA.ones(T, N)
Vs = (V, w)
result = CuArray{T}([0.0]) # We store the result in the GPU
result_unified = CuArray{T,1,CUDA.UnifiedMemory}([0.0]) # Or in the CPU (less efficient)


#%%
chosen_algo = "Dot"
chosen_N = 1000000

# Get the list of names for the given algorithm
# The keys of the dictionary for the chosen algo and type (e.g., Float32) are the names.
plot_names = ["Cublas", "Luma"]#names
mean_durations = [bench[Float64][chosen_algo][name][chosen_N]["mean_duration_gpu"] for name in plot_names]
bench[Float64][chosen_algo][name][chosen_N]
bench[Float64][chosen_algo]["Cublas"][1000000]