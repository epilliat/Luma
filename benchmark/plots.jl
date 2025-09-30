using Plots
using DataFrames

using Plots
using DataFrames

function create_kernel_stacked_barplot(df;
    algo="Dot",
    N=nothing,
    kernel_colors=[:blue, :red, :green, :orange, :purple, :cyan, :magenta, :yellow],
    overhead_color=:gray,
    overhead_alpha=0.7,
    kernel_labels=nothing)  # New parameter for custom labels
    # Filter for the specific algorithm and N
    if isnothing(N)
        df_filtered = filter(row -> row.algo == algo, df)
    else
        df_filtered = filter(row -> row.algo == algo && row.datalength == N, df)
    end

    if nrow(df_filtered) == 0
        if isnothing(N)
            error("No data found for algo='$algo'")
        else
            error("No data found for algo='$algo' and N=$N")
        end
    end

    # Get N value for display
    N_val = isnothing(N) ? df_filtered.datalength[1] : N
    N_str = N_val >= 1e9 ? "N=$(round(N_val/1e9, digits=1))e9" :
            N_val >= 1e6 ? "N=$(round(Int, N_val/1e6))e6" :
            N_val >= 1e3 ? "N=$(round(Int, N_val/1e3))e3" : "N=$N_val"

    names = sort(unique(df_filtered.name))
    datatypes = [Float32, Float64]

    # Initialize empty plot with N in title
    p = Plots.plot(
        legend=:topright,
        title="$algo: GPU Kernel Performance ($N_str)",
        ylabel="Duration (μs)",
        xlabel="",
        grid=true,
        framestyle=:box,
        bottom_margin=10 * Plots.Measures.pt
    )

    bar_width = 0.35
    bar_positions = Float64[]
    bar_labels = String[]
    max_height = 0.0

    # Dynamically find all kernel columns
    kernel_nums = Int[]
    for col in propertynames(df_filtered)
        col_str = String(col)
        if occursin(r"^kernel(\d+)$", col_str)
            m = match(r"^kernel(\d+)$", col_str)
            push!(kernel_nums, parse(Int, m.captures[1]))
        end
    end
    sort!(kernel_nums)
    max_kernels = isempty(kernel_nums) ? 0 : maximum(kernel_nums)

    # Check which kernels actually have data
    has_kernel = Dict{Int,Bool}()
    has_overhead = false

    # Try to get kernel names from the dataframe if available
    default_kernel_labels = Dict{Int,String}()
    for k in 1:max_kernels
        has_kernel[k] = false
        kernel_name_col = Symbol("kernel$(k)_name")

        # Check if there's a name column for this kernel
        if hasproperty(df_filtered, kernel_name_col)
            for row in eachrow(df_filtered)
                if !ismissing(row[kernel_name_col])
                    default_kernel_labels[k] = row[kernel_name_col]
                    break
                end
            end
        end
    end

    # Use custom labels if provided, otherwise use defaults or "Kernel N"
    final_kernel_labels = Dict{Int,String}()
    for k in 1:max_kernels
        if !isnothing(kernel_labels) && haskey(kernel_labels, k)
            final_kernel_labels[k] = kernel_labels[k]
        elseif haskey(default_kernel_labels, k)
            final_kernel_labels[k] = default_kernel_labels[k]
        else
            final_kernel_labels[k] = "Kernel $k"
        end
    end

    # First pass to check what components exist
    for name in names
        for dtype in datatypes
            subset = filter(row -> row.name == name && row.datatype == dtype, df_filtered)
            if nrow(subset) > 0
                row = subset[1, :]

                for k in 1:max_kernels
                    kernel_col = Symbol("kernel$k")
                    if hasproperty(row, kernel_col) && !ismissing(row[kernel_col]) && row[kernel_col] > 0
                        has_kernel[k] = true
                    end
                end

                overhead = row.median_duration_pipeline - row.mean_duration_gpu
                has_overhead = has_overhead || (overhead > 0)
            end
        end
    end

    # Track if we've added labels - initialize with all needed labels
    added_labels = Dict{String,Bool}()
    added_labels["CPU Overhead"] = false
    for k in 1:max_kernels
        added_labels[final_kernel_labels[k]] = false
    end

    # First pass: Add CPU Overhead to legend (but draw nothing yet)
    # This ensures it appears first in the legend
    dummy_rectangle = Shape([0, 0, 0, 0], [0, 0, 0, 0])
    if has_overhead
        Plots.plot!(p, dummy_rectangle, fillcolor=overhead_color, fillalpha=overhead_alpha,
            label="CPU Overhead", linecolor=overhead_color, linewidth=0)
        added_labels["CPU Overhead"] = true
    end

    # Second pass: Add all kernel labels in order (but draw nothing yet)
    for k in 1:max_kernels
        if has_kernel[k]
            color = kernel_colors[min(k, length(kernel_colors))]
            Plots.plot!(p, dummy_rectangle, fillcolor=color, fillalpha=1.0,
                label=final_kernel_labels[k], linecolor=color, linewidth=0)
            added_labels[final_kernel_labels[k]] = true
        end
    end

    # Now actually draw the bars
    position = 1.0
    for (group_idx, name) in enumerate(names)
        for dtype in datatypes
            push!(bar_positions, position)
            push!(bar_labels, dtype == Float32 ? "F32" : "F64")

            subset = filter(row -> row.name == name && row.datatype == dtype, df_filtered)

            if nrow(subset) > 0
                row = subset[1, :]

                # Build stack from bottom to top
                y_bottom = 0.0

                # Add all kernels dynamically
                for k in 1:max_kernels
                    kernel_col = Symbol("kernel$k")

                    if has_kernel[k] && hasproperty(row, kernel_col) && !ismissing(row[kernel_col]) && row[kernel_col] > 0
                        kernel_height = row[kernel_col]

                        # Choose color from the provided array
                        color = kernel_colors[min(k, length(kernel_colors))]

                        rectangle = Shape([position - bar_width, position - bar_width,
                                position + bar_width, position + bar_width],
                            [y_bottom, y_bottom + kernel_height,
                                y_bottom + kernel_height, y_bottom])

                        # Don't add label since we already added them above
                        Plots.plot!(p, rectangle, fillcolor=color, fillalpha=1.0,
                            label="", linecolor=color, linewidth=0.5)

                        y_bottom += kernel_height
                    end
                end

                # Overhead (top layer)
                overhead = row.median_duration_pipeline - row.mean_duration_gpu
                if overhead > 0
                    rectangle = Shape([position - bar_width, position - bar_width,
                            position + bar_width, position + bar_width],
                        [y_bottom, y_bottom + overhead,
                            y_bottom + overhead, y_bottom])
                    # Don't add label since we already added it above
                    Plots.plot!(p, rectangle, fillcolor=overhead_color, fillalpha=overhead_alpha,
                        label="", linecolor=overhead_color, linewidth=0.5)
                    y_bottom += overhead
                end

                max_height = max(max_height, y_bottom)
            end

            position += 1.0
        end
        position += 0.5  # Gap between groups
    end

    # Set axis limits and ticks
    Plots.plot!(p,
        xticks=(bar_positions, bar_labels),
        xlims=(0.2, maximum(bar_positions) + 0.8),
        ylims=(0, max_height * 1.1))

    # Add group labels BETWEEN the bars of each group
    for (i, name) in enumerate(names)
        # Position label between F32 and F64 bars of each group
        if i == 1
            label_x = 1.5  # Between positions 1 and 2
        else
            label_x = 1.5 + (i - 1) * 2.5  # Account for gap between groups
        end

        annotate!(p, label_x, -max_height * 0.1,
            text(name, 10, :center))
    end

    return p
end

cub_f32 = DataFrame(
    name="Cub",
    datatype=Float32,
    algo="Sum",
    datalength=bench.datalength[1],
    kernel1=0.0,
    kernel2=0.0,
    kernel3=0.0,
    kernel4=11.0,  # NVCC benchmark time
    kernel1_acc=0.0,
    kernel2_acc=0.0,
    kernel3_acc=0.0,
    kernel4_acc=11.0,
    kernel4_name="NVCC Benchmark",  # This will be used as the label
    mean_duration_gpu=11.0,
    median_duration_pipeline=11.0,
    # ... other columns
)

cub_f64 = DataFrame(
    name="Cub",
    datatype=Float64,
    algo="Sum",
    datalength=bench.datalength[1],
    kernel1=0.0,
    kernel2=0.0,
    kernel3=0.0,
    kernel4=25.0,
    kernel1_acc=0.0,
    kernel2_acc=0.0,
    kernel3_acc=0.0,
    kernel4_acc=25.0,
    kernel4_name="NVCC Benchmark",
    mean_duration_gpu=25.0,
    median_duration_pipeline=25.0,
    # ... other columns
)
bench.kernel1_name .= missing
bench.kernel2_name .= missing
bench_with_cub = vcat(bench, cub_f32, cub_f64, cols=:union)

# Plot - it will automatically use "NVCC Benchmark" from kernel4_name
plot = create_kernel_stacked_barplot(bench_with_cub,
    algo="L2 norm",
    kernel_colors=[:blue, :red, :green, :orange], overhead_alpha=1)

# Usage examples:
# plot = create_kernel_stacked_barplot(bench, algo="Dot")
# plot = create_kernel_stacked_barplot(bench, algo="Dot", kernel_colors=[:purple, :orange, :pink])
# plot = create_kernel_stacked_barplot(bench, algo="Dot", overhead_color=:black, overhead_alpha=0.5)
# display(plot)