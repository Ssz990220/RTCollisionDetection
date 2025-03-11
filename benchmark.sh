# If the folder does not exist, create one
mkdir -p ./data/Benchmark/result/discrete
mkdir -p ./data/Benchmark/result/curve
# Execute the scripts, takes about few minutes
./build/bin/Benchmarks/benchmarkDenseShelf --benchmark_out=./data/Benchmark/result/discrete/denseShelf.json
./build/bin/Benchmarks/benchmarkShelf --benchmark_out=./data/Benchmark/result/discrete/shelf.json
./build/bin/Benchmarks/benchmarkShelfSimple --benchmark_out=./data/Benchmark/result/discrete/shelfSimple.json
./build/bin/Benchmarks/BM_CURVE_DenseShelf --benchmark_out=./data/Benchmark/result/curve_loop/cur_denseShelf.json
./build/bin/Benchmarks/BM_CURVE_Shelf --benchmark_out=./data/Benchmark/result/curve_loop/cur_shelf.json
./build/bin/Benchmarks/BM_CURVE_ShelfSimple --benchmark_out=./data/Benchmark/result/curve_loop/cur_shelfSimple.json