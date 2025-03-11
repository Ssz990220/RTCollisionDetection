@echo off
SET "baseDir=.\build\bin\Benchmarks\Release"
SET "outputDir=.\data\Benchmark\result\discrete"

:: Check if the output directory exists, create it if it does not
IF NOT EXIST "%outputDir%" (
    echo Output directory not found. Creating %outputDir%...
    mkdir "%outputDir%"
)

:: Define each executable and corresponding output file
SETLOCAL EnableDelayedExpansion
SET "exe[1]=benchmarkDenseShelf.exe --benchmark_out=%outputDir%\denseShelf.json"
SET "exe[2]=benchmarkShelf.exe --benchmark_out=%outputDir%\shelf.json"
SET "exe[3]=benchmarkShelfSimple.exe --benchmark_out=%outputDir%\shelfSimple.json"

SET "outputDir=.\data\Benchmark\result\discrete"

:: Check if the output directory exists, create it if it does not
IF NOT EXIST "%outputDir%" (
    echo Output directory not found. Creating %outputDir%...
    mkdir "%outputDir%"
)

SET "exe[4]=BM_CURVE_DenseShelf.exe --benchmark_out=%outputDir%\cur_denseShelf.json"
SET "exe[5]=BM_CURVE_Shelf.exe --benchmark_out=%outputDir%\cur_shelf.json"
SET "exe[6]=BM_CURVE_ShelfSimple.exe --benchmark_out=%outputDir%\cur_shelfSimple.json"

:: Execute each script
FOR /L %%G IN (1,1,6) DO (
    SET "cmd=!exe[%%G]!"
    echo Executing: %baseDir%\!cmd!
    call %baseDir%\!cmd!
)

echo All benchmarks executed.
pause