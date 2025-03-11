begin
    using Pkg
    Pkg.activate("./scripts")
    using StaticArrays, LinearAlgebra, JSON, DataFrames, YAML, FilePathsBase, Statistics, Printf
    # using GLMakie
    using CairoMakie
    import Base: Regex
end

# define plot color
clrs = Makie.to_colormap(Makie.wong_colors())
clr_two_way = Makie.to_colormap(:Paired_7)[6]
push!(clrs, clr_two_way)

# Function to parse and flatten the JSON data
function flatten_json(json_data)
    for (robotName, scales) in json_data
        for (baseScale, samples) in scales
            for (nSample, traj_ids) in samples
                for (traj_id, details) in traj_ids
                    for (key, metrics) in details
                        # Extract metrics
                        diff = metrics["Diff"]
                        gt = metrics["GT"]
                        intersection = metrics["Intersection"]
                        miss = metrics["Miss"]
                        precision = metrics["Precision"]
                        recall = metrics["Recall"]
                        rep = metrics["Rep"]
                        voxel_size = metrics["voxel size"]
                        if !haskey(metrics, "hausdorff")
                            hausdorff = 0.0
                        else
                            hausdorff = metrics["hausdorff"]
                        end

                        # Append the data to the DataFrame
                        push!(df, (robotName, parse(Int, baseScale), parse(Int, nSample), parse(Int, traj_id),
                            key, diff, gt, intersection, miss, precision, recall, rep, voxel_size, hausdorff))
                    end
                end
            end
        end
    end
end

function parse_string(s)
    regex = r"^(\d+)_(\d+)_(\d+)"
    match_result = match(regex, s)
    type = parse(Int32, match_result[1])
    nCPts = parse(Int32, match_result[2])
    nTPts = parse(Int32, match_result[3])
    val = type * 1e4 + nCPts * 1e2 + nTPts
    return val
end

function resultOneSample(nSamples, df; disable_mask=[], disabled_traj=[])
    df_nSample = df[df.nSample.==nSamples, :]
    df_nSample = df_nSample[df_nSample.traj_id.∉[disabled_traj], :]
    # average over all traj_id
    df_avg = combine(groupby(df_nSample, [:type]), :Precision => Statistics.mean => :Precision, :Recall => Statistics.mean => :Recall)

    show_type = unique(df_avg.type)
    # drop 2_4_16 & 2_8_32
    show_type = show_type[show_type.∉[disable_mask]]
    show_name = [replace(s, r"_\d+$" => "") for s in show_type]
    # create a dict that maps shot_type to show_name
    show_name_dict = Dict(zip(show_type, show_name))

    df_mean_show = df_avg[df_avg.type.∈[show_type], :]
    df_mean_show = sort(df_mean_show, :type, by=x -> parse_string(x))
    @show df_mean_show


    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1],
        xlabel="Precision", ylabel="Recall",
        limits=(65, 103, 37, 103),
        title="Precision vs Recall for nSample" * string(nSamples),
        titlesize=22)
    types = unique(df_mean_show.type)
    types = sort(types, by=x -> parse_string(x))
    for type in types
        df_type = df_mean_show[df_mean_show.type.==type, :]
        plot!(ax, df_type.Precision, df_type.Recall, label=type)
        if df_type.Precision[1] < 80
            text!(type, position=(df_type.Precision[1], df_type.Recall[1]), align=(:right, :bottom))
        end
    end
    axislegend(position=:cb, titlesize=22, margin=(200, 0, 0, 0), nbanks=2)
    fig
end

function readYml(filename)
    data = open(filename) do f
        YAML.load(f)
    end
    return data
end

function parse_benchmark_string(s::String)
    regex = r"BM_([A-Z]+)<(\d+)(?:, (\d+))?(?:, (\d+))?>/(\d+)/min_warmup_time:\d+\.\d+/real_time"
    match_result = match(regex, s)
    if match_result === nothing
        error("String does not match the expected format")
    end
    name = match_result[1]
    nCPts = parse(Int32, match_result[2])
    if match_result[3] === nothing
        nTPts = 0
    else
        nTPts = parse(Int32, match_result[3])
    end
    nPoses = parse(Int32, match_result[5])

    return name, nCPts, nTPts, nPoses
end

function process_result(result)
    s = result["name"]
    t = result["real_time"]
    if t == 0
        t = NaN
    end
    name, nCPts, nTPts, nPoses = parse_benchmark_string(s)
    return name, nCPts, nTPts, nPoses, t
end

function buildLegend(f, fidx; margin=(0, 0, 0, 0), halign=:right, valign=:top)
    elem_32 = [LineElement(color=:black, linestyle=:dashdot, linewidth=3)]
    elem_16 = [LineElement(color=:black, linestyle=:solid, linewidth=3)]
    elem_8 = [LineElement(color=:black, linestyle=:dash, linewidth=3)]
    elem_4 = [LineElement(color=:black, linestyle=:dot, linewidth=3)]

    elem_mesh = [PolyElement(color=clrs[8])]
    elem_lin = [PolyElement(color=clrs[1])]
    elem_quad = [PolyElement(color=clrs[3])]
    elem_curobo = [PolyElement(color=clrs[6])]

    legend = Legend(f[fidx...],
        [[elem_4, elem_8, elem_16, elem_32], [elem_curobo, elem_mesh, elem_lin, elem_quad]],
        [["4", "8", "16", "32"], ["cuRobo", "Two-way", "Linear", "Quadratic"]],
        ["Number of\nControl\nPoints", "Method"],
        margin=margin,
        tellheight=false,
        tellwidth=false,
        halign=halign,
        valign=valign)
    legend.nbanks = 2
    legend.titleposition = :left
end


function plotCurBM(RT_filename, curoboFile, outputFile, t_cpu_per_pose)
    data = open(RT_filename) do f
        JSON.parse(read(f, String))
    end

    results = data["benchmarks"]

    df = DataFrame(process_result.(results), [:Name, :nCPts, :nTPts, :nPoses, :Time])

    # replace BM_LINEAR with Lin, BM_QUAD with Quad
    df.Name = replace.(df.Name, "LINEAR" => "Lin", "QUAD" => "Quad", "MESH" => "Mesh", "RAY" => "Ray")

    mesh_rows = filter(row -> row.Name == "Mesh", df)
    ray_rows = filter(row -> row.Name == "Ray", df)

    # Ensure the rows are aligned by nCPts and nPoses
    combined_df = DataFrame()

    for (i, mesh_row) in enumerate(eachrow(mesh_rows))
        ray_row = ray_rows[i, :]
        if mesh_row.nCPts == ray_row.nCPts && mesh_row.nPoses == ray_row.nPoses
            new_row = DataFrame(Name="Two-Way", nCPts=mesh_row.nCPts, nTPts=mesh_row.nTPts, nPoses=mesh_row.nPoses, Time=mesh_row.Time + ray_row.Time)
            combined_df = vcat(combined_df, new_row)
        end
    end

    df = vcat(df, combined_df)

    # now average the time for each (Name, nCPts, nTPts, nPoses) combination
    # unite name nCPts nTPts into a single colume Name_nCPts_nTPts
    df_mean = df
    df_mean[!, :type] = [name * "-" * string(nCPts) for (name, nCPts) in zip(df_mean.Name, df_mean.nCPts)]

    # move type to the first column and drop Name, nCPts, nTPts
    df_mean = df_mean[:, [:type, :nCPts, :nPoses, :Time]]

    # process the curobo result
    curoboResult = readYml(curoboFile)

    # convert the curobo result to a DataFrame
    df_curobo = DataFrame(curoboResult)
    # replace colume names: Batch Size => nPoses, Collision Checking => Time, Traj Downsample => nCPts
    rename!(df_curobo, Dict("Batch Size" => :nPoses, "Collision Checking" => :Time, "sweep_steps" => :nCPts))
    df_curobo

    # add a colume called Name and fill it with curobo
    df_curobo[!, :Name] = ["CUROBO" for i in 1:size(df_curobo, 1)]
    # drop rows whose nPoses is smaller than 512
    # df_curobo = df_curobo[df_curobo.nSamples.>=512, :]

    # now average the time for each (Name, nCPts, nTPts, nPoses) combination
    df_curobo_mean = combine(groupby(df_curobo, [:Name, :nCPts, :nPoses]), :Time => Statistics.mean => :Time)

    # unite name nCPts nTPts into a single colume Name_nCPts_nTPts
    df_curobo_mean[!, :type] = [name * "-" * string(nCPts) for (name, nCPts) in zip(df_curobo_mean.Name, df_curobo_mean.nCPts)]

    # move type to the first column and drop Name, nCPts, nTPts
    df_curobo_mean = df_curobo_mean[:, [:type, :nCPts, :nPoses, :Time]]

    # convert the unit of time in df_curobo_mean from second to microsecond
    df_curobo_mean.Time = df_curobo_mean.Time .* 1e6

    # combine the two DataFrames
    df_combined = vcat(df_mean, df_curobo_mean)

    linear_rows = filter(row -> occursin("Lin", row.type), df_combined)
    curobo_rows = filter(row -> (occursin("CUROBO", row.type) && !(occursin("CUROBO-4", row.type))), df_combined)

    # sort curobo_rows by type, then by nPoses
    linear_rows = sort(linear_rows, [:type, :nPoses])
    curobo_rows = sort(curobo_rows, [:type, :nPoses])

    nCompare = size(unique(curobo_rows.nPoses))[1]
    nTypes = size(unique(curobo_rows.type))[1]
    speedUp = zeros(nTypes, nCompare)

    # divide the time of curobo by linear to get the speedup
    for (i, linear_row) in enumerate(eachrow(linear_rows))
        # @show linear_row
        row_idx = findfirst(x -> x == linear_row.nCPts, unique(linear_rows.nCPts))
        column_idx = findfirst(x -> x == linear_row.nPoses, unique(linear_rows.nPoses))
        curobo_row = curobo_rows[i, :]
        speedUp[row_idx, column_idx] = curobo_row.Time / linear_row.Time
    end

    for nCPts in unique(linear_rows.nCPts)
        @printf("For %d cPts, the max speed up is %.4f, the average speed up is %.4f\n", nCPts, maximum(speedUp[findall(x -> x == nCPts, unique(linear_rows.nCPts)), :]), mean(speedUp[findall(x -> x == nCPts, unique(linear_rows.nCPts)), :]))
    end


    # plot the result, the plot shows how each type's time changes with nPoses
    begin
        fig = Figure(size=(600, 350))
        x_labels = string.(unique(df_combined.nPoses))

        # compute the average time for each pose by dividing the time by nPoses
        df_combined[!, :TimePerPose] = df_combined.Time ./ df_combined.nPoses

        # add cpu_result to df_combined
        for n in x_labels
            push!(df_combined, ["CPU-16", 16, parse(Int, n), t_cpu_per_pose * 16 * parse(Int, n), t_cpu_per_pose * 16])
        end

        # find the fastest Lin-16
        lin_df = filter(row -> row.type == "Lin-16", df_combined)
        fastest_lin = minimum(lin_df[!, "TimePerPose"])
        @printf("RT Method is up to %f times faster than CPU Method.", t_cpu_per_pose * 16 / fastest_lin)

        for n in x_labels
            push!(df_combined, ["CPU-32", 32, parse(Int, n), t_cpu_per_pose * 32 * parse(Int, n), t_cpu_per_pose * 32])
        end


        #define the x-ticks by unique nPoses
        ax = Axis(fig[1, 1],
            xticks=(2 .^ (0:length(x_labels)-1), x_labels),
            xlabel="Number of Trajectories per Batch", ylabel="Time [μs / trajectory]",
            xlabelsize=24, ylabelsize=24,
            xticklabelrotation=pi / 4,
            xscale=log2, yscale=log10)
        for type in unique(df_combined.type)
            df_type = df_combined[df_combined.type.==type, :]

            if occursin("CUROBO", type)
                color = clrs[6]
            elseif occursin("CPU", type)
                color = :black
            elseif occursin("Two-Way", type)
                color = clrs[8]
            elseif occursin("Lin", type)
                color = clrs[1]
            elseif occursin("Quad", type)
                color = clrs[3]
            else
                continue
            end

            if occursin("4", type)
                linestyle = :dashdot
            elseif occursin("8", type)
                linestyle = :dash
            elseif occursin("16", type)
                linestyle = :solid
            elseif occursin("32", type)
                linestyle = :dot
            else
                linestyle = :solid
            end

            if occursin("4", type) && occursin("CUROBO", type)
                continue
            end

            lines!(df_type.nPoses, df_type.TimePerPose, label=type, color=color, linestyle=linestyle, linewidth=3)
        end
        buildLegend(fig, (1, 1))
    end
    save(outputFile * ".pdf", fig)
    save(outputFile * ".svg", fig)
    save(outputFile * ".png", fig)

    return fig
end