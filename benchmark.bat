@echo off
:: Runs the collision-checking benchmarks for the three shelf scenes and writes Google Benchmark JSON where the
:: plotting scripts read it (scripts\Benchmark\plots\{discrete,curve}):
::   <out>\discrete\{denseShelf,shelf,shelfSimple}.json
::   <out>\curve\{cur_denseShelf,cur_shelf,cur_shelfSimple}.json
:: Takes a few minutes per executable. Run from the repository root after building.
::
:: usage: benchmark.bat [--legacy] [--only discrete or curve] [--build <dir>] [--out <dir>] [--no-pause] ["<GB arg>" ...]
::   (default)   new stack: build\bin\apps\rtcd_bench_discrete.exe and rtcd_bench_curve.exe, scene chosen with
::               --scene (build targets rtcd_bench_discrete rtcd_bench_curve, part of AllNew)
::   --legacy    old stack (RTCD\, frozen; deleted with it): build\bin\Benchmarks\benchmarkShelf.exe ...
::               BM_CURVE_*.exe (build target AllBenchmarks)
::   --only      run only the discrete or only the curve benchmarks
::   --build     build directory (default .\build); executables are looked up in <build>\bin\<apps|Benchmarks>
::               (Ninja, single-config) and <build>\bin\<apps|Benchmarks>\Release (Visual Studio generator)
::   --out       result root (default .\data\Benchmark\result)
::   --no-pause  do not wait for a key at the end
::   any other argument is passed to every executable; quote arguments that contain '=' (cmd splits on it),
::   e.g. "--benchmark_filter=BM_IAS"
:: Both stacks produce the same benchmark names, so the plotting scripts read either. Before Stage 3 the curve
:: results went to <out>\curve_loop\, which the plotting scripts do not read; they now go to <out>\curve\.
setlocal EnableDelayedExpansion

set "LEGACY=0"
set "ONLY=all"
set "BUILD=.\build"
set "OUT=.\data\Benchmark\result"
set "NOPAUSE=0"
set "EXTRA="

:parse
if "%~1"=="" goto parsed
if /I "%~1"=="--legacy"   ( set "LEGACY=1"   & shift & goto parse )
if /I "%~1"=="--no-pause" ( set "NOPAUSE=1"  & shift & goto parse )
if /I "%~1"=="--only"     ( set "ONLY=%~2"   & shift & shift & goto parse )
if /I "%~1"=="--build"    ( set "BUILD=%~2"  & shift & shift & goto parse )
if /I "%~1"=="--out"      ( set "OUT=%~2"    & shift & shift & goto parse )
set "ARG=%~1"
set "EXTRA=!EXTRA! "!ARG!""
shift
goto parse
:parsed

if /I not "%ONLY%"=="all" if /I not "%ONLY%"=="discrete" if /I not "%ONLY%"=="curve" (
    echo --only must be discrete or curve
    exit /b 2
)

if "%LEGACY%"=="1" ( set "SUB=Benchmarks" & set "PROBE=benchmarkShelf.exe" ) else ( set "SUB=apps" & set "PROBE=rtcd_bench_discrete.exe" )
set "BIN="
if exist "%BUILD%\bin\%SUB%\%PROBE%" set "BIN=%BUILD%\bin\%SUB%"
if not defined BIN if exist "%BUILD%\bin\%SUB%\Release\%PROBE%" set "BIN=%BUILD%\bin\%SUB%\Release"
if not defined BIN (
    echo No %PROBE% in %BUILD%\bin\%SUB% or %BUILD%\bin\%SUB%\Release; build it first.
    exit /b 1
)

for %%D in (discrete curve) do (
    if not exist "%OUT%\%%D" (
        echo Output directory not found. Creating %OUT%\%%D...
        mkdir "%OUT%\%%D"
    )
)

:: Each entry: executable, output file (relative to --out), scene argument (new stack only)
set "N=0"
if "%LEGACY%"=="0" (
    if /I not "%ONLY%"=="curve" (
        call :add rtcd_bench_discrete.exe "discrete\denseShelf.json" "--scene shelf_dense"
        call :add rtcd_bench_discrete.exe "discrete\shelf.json" "--scene shelf"
        call :add rtcd_bench_discrete.exe "discrete\shelfSimple.json" "--scene shelf_simple"
    )
    if /I not "%ONLY%"=="discrete" (
        call :add rtcd_bench_curve.exe "curve\cur_denseShelf.json" "--scene shelf_dense"
        call :add rtcd_bench_curve.exe "curve\cur_shelf.json" "--scene shelf"
        call :add rtcd_bench_curve.exe "curve\cur_shelfSimple.json" "--scene shelf_simple"
    )
) else (
    if /I not "%ONLY%"=="curve" (
        call :add benchmarkDenseShelf.exe "discrete\denseShelf.json" ""
        call :add benchmarkShelf.exe "discrete\shelf.json" ""
        call :add benchmarkShelfSimple.exe "discrete\shelfSimple.json" ""
    )
    if /I not "%ONLY%"=="discrete" (
        call :add BM_CURVE_DenseShelf.exe "curve\cur_denseShelf.json" ""
        call :add BM_CURVE_Shelf.exe "curve\cur_shelf.json" ""
        call :add BM_CURVE_ShelfSimple.exe "curve\cur_shelfSimple.json" ""
    )
)

set "FAILED=0"
for /L %%G in (1,1,%N%) do (
    set "cmd="%BIN%\!exe[%%G]!" !args[%%G]! --benchmark_out="%OUT%\!json[%%G]!" --benchmark_out_format=json!EXTRA!"
    echo Executing: !cmd!
    !cmd!
    if errorlevel 1 (
        echo FAILED: !exe[%%G]! ^(exit !errorlevel!^)
        set "FAILED=1"
    )
)

if "%FAILED%"=="1" ( echo Some benchmarks failed. ) else ( echo All benchmarks executed; results in %OUT%\discrete and %OUT%\curve. )
if "%NOPAUSE%"=="0" pause
if "%FAILED%"=="1" exit /b 1
exit /b 0

:add
set /a N+=1
set "exe[%N%]=%~1"
set "json[%N%]=%~2"
set "args[%N%]=%~3"
exit /b 0
