include("resultAnalysisUtils.jl")
set_theme!(fonts=(; regular="Helvetica", bold="Helvetica Bold"))

inputDir = "./data/Benchmark/result/discrete/"
plotDir = "./data/Benchmark/Plots/discrete/"
# if plotDir does not exist, create it
if !isdir(plotDir)
    mkpath(plotDir)
end

show_fig = false;
begin
    RT_fileName = inputDir * "denseShelf.json"
    curoboFile = inputDir * "kinematics_benchmark_shelf_dense.yml"
    fig = plotResult(RT_fileName, curoboFile, 105000 / 4096)
    save(plotDir * "DenseShelf.pdf", fig)
    save(plotDir * "DenseShelf.svg", fig)
    save(plotDir * "DenseShelf.png", fig)
    if show_fig
        fig
    end
end

begin
    RT_fileName = inputDir * "shelf.json"
    curoboFile = inputDir * "kinematics_benchmark_shelf.yml"
    curoboResult = readYml(curoboFile)
    fig = plotResult(RT_fileName, curoboFile, 95000 / 4096)
    save(plotDir * "Shelf.pdf", fig)
    save(plotDir * "Shelf.svg", fig)
    save(plotDir * "Shelf.png", fig)
    if show_fig
        fig
    end
end

begin
    RT_fileName = inputDir * "shelfSimple.json"
    curoboFile = inputDir * "kinematics_benchmark_shelf_simple.yml"
    curoboResult = readYml(curoboFile)
    fig = plotResult(RT_fileName, curoboFile, 70000 / 4096)
    save(plotDir * "ShelfSimple.pdf", fig)
    save(plotDir * "ShelfSimple.svg", fig)
    save(plotDir * "ShelfSimple.png", fig)
    if show_fig
        fig
    end
end