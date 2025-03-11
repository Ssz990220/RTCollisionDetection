include("resultAnalysisUtils.jl")

inputDir = "./data/Benchmark/result/curve/"
outputDir = "./data/Benchmark/Plots/curve/"
set_theme!(fonts=(; regular="Helvetica", bold="Helvetica Bold"))

# create dir if not exists
if !isdir("./data/Benchmark/result/curve")
    mkpath("./data/Benchmark/result/curve")
end
if !isdir(outputDir)
    mkpath(outputDir)
end
begin
    RT_filename = inputDir * "cur_denseShelf.json"
    curoboFile = inputDir * "curobo_swept_benchmark_shelf_dense.yml"
    outputFile = outputDir * "cur_denseShelf"
    plotCurBM(RT_filename, curoboFile, outputFile, 4060 * 1000 / 4096 / 32)
end

begin
    RT_filename = inputDir * "cur_shelf.json"
    curoboFile = inputDir * "curobo_swept_benchmark_shelf.yml"
    outputFile = outputDir * "cur_shelf"
    plotCurBM(RT_filename, curoboFile, outputFile, 4065 * 1000 / 4096 / 32)
end

begin
    RT_filename = inputDir * "cur_shelfSimple.json"
    curoboFile = inputDir * "curobo_swept_benchmark_shelf_simple.yml"
    outputFile = outputDir * "cur_shelfSimple"
    plotCurBM(RT_filename, curoboFile, outputFile, 2938 * 1000 / 4096 / 32)
end