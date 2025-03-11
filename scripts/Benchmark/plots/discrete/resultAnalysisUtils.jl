begin
    using Pkg
    Pkg.activate("./scripts")
    using StaticArrays, LinearAlgebra, JSON, DataFrames, YAML, FilePathsBase
    using CairoMakie
    import Base: Regex
end

function readYml(filename)
    data = open(filename) do f
        YAML.load(f)
    end
    return data
end

function parse_benchmark_string(s::String)
    # regex = r"^(.+)/(\d+)/+$"
    regex = r"^([^/]+)/(\d+)/"
    match_result = match(regex, s)
    if match_result === nothing
        error("String does not match the expected format")
    end
    name = match_result[1]
    # batchsize = parse(Int32, match_result[2])
    nPoses = parse(Int32, match_result[2])
    return name, nPoses
end

function process_result(result)
    s = result["run_name"]
    t = result["real_time"]
    if t == 0
        t = NaN
    end
    name, nPoses = parse_benchmark_string(s)
    return name, nPoses, t
end


function rank_column(col)
    filtered_col = filter(!isnan, col)
    max_val = isempty(filtered_col) ? NaN : maximum(filtered_col)
    min_val = isempty(filtered_col) ? NaN : minimum(filtered_col)
    normalized_col = (max_val .- col) ./ (max_val - min_val)
    return normalized_col
end

function plotResult(RT_fileName, curoboFile, cpu_t, selected=true)
    data = open(RT_fileName) do f
        JSON.parse(read(f, String))
    end
    begin
        results = data["benchmarks"]

        df = DataFrame(process_result.(results), [:Name, :nPoses, :Time])
        df[!, :NameBatchsize] = df.Name

        df_pivoted = unstack(df, :NameBatchsize, :nPoses, :Time)
        curoboResult = readYml(curoboFile)
        # convert the curobo result to a DataFrame
        df_curobo = DataFrame(curoboResult)
        # drop robot
        df_curobo = select!(df_curobo, Not(:robot))
        # rename colume names: Batch Size => nPoses, Collision Checking => Time
        rename!(df_curobo, Dict("Batch Size" => :nPoses, "Collision Checking" => :Time))

        # create a new df with df_curobo, use "Batch Size" as the name of each row, and "Collision Checking" as the value
        df_transformed = DataFrame(NameBatchsize=["CUROBO"])
        row_data = permutedims(df_curobo.Time)
        for (i, batch_size) in enumerate(df_curobo.nPoses)
            df_transformed[!, string(batch_size)] = [row_data[i]] * 1e6
        end

        # concatenate the two DataFrames
        df_pivoted = vcat(df_transformed, df_pivoted)

        if selected
            # drop rows that have 256, 512 in its NameBatchsize
            df_pivoted = df_pivoted[.!occursin.("256", df_pivoted.NameBatchsize).&.!occursin.("512", df_pivoted.NameBatchsize), :]
        end

        # replace name BM_RAY_ONLINE_UNI_128 with RAY, BM_IAS_UNI_128 with IAS, BM_GAS_UNI_128 with GAS
        # df_pivoted.NameBatchsize = replace.(df_pivoted.NameBatchsize, "BM_RAY_ONLINE_UNI-128" => "RobotToObs", "BM_IAS_UNI-128" => "ObsToRobot", "BM_GAS_UNI-128" => "GAS_SPHR", "BM_IAS_SPHR_UNI-128" => "IAS_SPHR")
        df_pivoted.NameBatchsize = replace.(df_pivoted.NameBatchsize, "BM_RAY" => "RobotToObs", "BM_IAS" => "ObsToRobot")

        # sort the columns based on the the column name
        df_pivoted = df_pivoted[:, ["NameBatchsize", (sort(names(df_pivoted)[2:end], by=x -> parse(Int, x)))...]]
    end

    # add df_curobo which is a colume as a row to df_pivoted
    begin

        data_matrix = Matrix(df_pivoted[:, 2:end])  # Convert selected DataFrame portion to a Matrix

        rank_matrix = mapslices(rank_column, data_matrix, dims=1)
    end
    # Now plot
    fig = Figure(size=(600, 350))
    begin
        # Extracting names for y-axis labels
        y_labels = df_pivoted.NameBatchsize

        # Extracting nPoses for x-axis labels (assuming columns of df_pivoted after the first are nPoses)
        x_labels = names(df_pivoted)[2:end] # Skip the first column which is NameBatchsize
    end

    begin
        # Use format string to display the title
        cpu_r = string(round(cpu_t, digits=2))
        # compute the per pose Time

        # Initialize the new DataFrame with the NameBatchsize column
        df_avg = DataFrame(NameBatchsize=df_pivoted.NameBatchsize)

        # Iterate over the columns (excluding the first column)
        for i in 2:size(df_pivoted, 2)
            col = parse(Float64, names(df_pivoted)[i])
            df_avg[!, string(names(df_pivoted)[i])] = df_pivoted[:, i] ./ col
        end
        df_avg

        # convert the dataframe to a matrix
        data_matrix = Matrix(df_avg[:, 2:end])
        # use log to scale the data for better viz
        data_matrix_log = log.(data_matrix)

        speedUp = Vector(df_avg[1, 2:end]) ./ (Vector(df_avg[2, 2:end]) .+ Vector(df_avg[3, 2:end]))
        @show speedUp
        maxSpeedup = maximum(speedUp)
        @show maxSpeedup

        speedUp = [cpu_t] ./ (Vector(df_avg[2, 2:end]) .+ Vector(df_avg[3, 2:end]))
        @show speedUp
        maxSpeedup = maximum(speedUp)
        @show maxSpeedup
    end

    begin
        # drop rows with NaN in data_matrix
        line_data_matrix = data_matrix
        cpu_ = [cpu_t for i in 1:size(data_matrix, 2)]
        line_data_matrix = [cpu_'; line_data_matrix]
        # extract name based on row_idx
        line_names = df_pivoted.NameBatchsize
        line_names = ["CPU" line_names...]

        # extract the index for "ObsToRobot" and "RobotToObs" from line_names
        idx_ray = findall(x -> x == "RobotToObs", line_names)[1][2]
        idx_ias = findall(x -> x == "ObsToRobot", line_names)[1][2]

        # sum the time for "RobotToObs" and "ObsToRobot"
        ray_time = sum(line_data_matrix[[idx_ray, idx_ias], :], dims=1)

        # concatenate the sum to line_data_matrix and name to line_names
        line_names = [line_names..., "Two-way"]
        line_data_matrix = [line_data_matrix; ray_time]


        clrs = Makie.to_colormap(:Set1_7)

        ax3 = Axis(fig[1, 1],
            xticks=(1:length(x_labels), x_labels),
            xlabel="Number of Poses in Batch", ylabel="Time [μs / pose]",
            yscale=log10,
            xticklabelrotation=pi / 4,
            xlabelsize=24, ylabelsize=24)
        for i in 1:size(line_data_matrix, 1)
            label = line_names[i]
            if occursin("RobotToObs", label) && !occursin("Two-way", label)
                clr = clrs[2]
                linestyle = :solid
            elseif occursin("IAS_SPHR", label)
                clr = clrs[4]
                linestyle = :dash
            elseif occursin("ObsToRobot", label) && !occursin("IAS_SPHR", label) && !occursin("Two-way", label)
                clr = clrs[3]
                linestyle = :solid
            elseif occursin("CUROBO", label)
                clr = Makie.to_colormap(Makie.wong_colors())[6]
                # clr = clrs[6]
                linestyle = :dash
            elseif occursin("GAS", label)
                clr = clrs[7]
                linestyle = :solid
            elseif occursin("Two-way", label)
                clr = :red
                linestyle = :solid
            else
                clr = :black
                linestyle = :dash
            end

            lines!(ax3, line_data_matrix[i, :], label=label, linewidth=4, linestyle=linestyle, color=clr)
        end
        # show label
        axislegend(ax3)
        fig
    end
end
