# RTCollisionDetection
This package implements mesh-to-mesh and mesh-to-robot-sphere-swept-volume collision detection algorithms using [NVIDIA OptiX](https://developer.nvidia.com/rtx/ray-tracing/optix).

<table align="center">
  <tr>
    <td><img src="assets/Headline_Disc_blender.png" width="100%" /></td>
    <td><img src="assets/Headline_Ray_blender.png" width="100%" /></td>
    <td><img src="assets/Headline_Lin_blender.png" width="100%" /></td>
    <td><img src="assets/Headline_Quad_blender.png" width="100%" /></td>
  </tr>
  <tr>
    <td align="center">(a)</td>
    <td align="center">(b)</td>
    <td align="center">(c)</td>
    <td align="center">(d)</td>
  </tr>
</table>

<p align="center">
  <em>Ray tracing collision detection methods: discrete-pose collision detection by ray-tracing (a) along obstacle meshes and (b) along robot meshes, and continuous collision detection by ray-tracing against swept sphere-approximated robot volumes (c) piecewise-linear paths or (d) quadratic B-spline paths.</em>
</p>


## TODO:
- [x] Upload the source code.
- [ ] URDF parser. (How to setup a new robot)
- [ ] Merge RobotToObs and ObsToRobot (Two-way method)
- [ ] Upload modified GVDB.
- [ ] Enable self-collision
- [ ] Adapt curve representation for Blackwell Architecture

## Features:
### **Mesh-to-mesh** collision detection with ray-tracing method.

<table align="center">
  <tr>
    <td><img src="assets/RTBunny_1.gif" width="100%" /></td>
    <td><img src="assets/RTBunny_3.gif" width="100%" /></td>
  </tr>
  <tr>
    <td align="center">ObsToRobot</td>
    <td align="center">RobotToObs</td>
  </tr>
</table>

<table align="center">
  <tr>
    <td><img src="assets/DenseShelf_4070.jpg" width="100%" /></td>
    <td><img src="assets/Shelf_4070.jpg" width="100%" /></td>
    <td><img src="assets/ShelfSimple_4070.jpg" width="100%" /></td>
  </tr>
  <tr>
    <td align="center">Dense Scene</td>
    <td align="center">Medium Scene</td>
    <td align="center">Simple Scene</td>
  </tr>
</table>
<p align="center">
  <em>Despite performing a much more complex collision check than cuRobo</em> (<b>mesh-to-mesh vs. mesh-to-sphere</b>), <em>our methods outpaced cuRobo by</em> <b>up to 2.8x</b> <em>on medium to dense scenes, scaling past CPU performance at smaller batch sizes.</em>
</p>

### **Swept-volume of sphere represented robot** collision deteciton with mesh obstacles.
<p align="center">
  <img src="assets/CurveGen.gif" width="96%" />
</p>
<table align="center">
  <tr>
    <td><img src="assets/cur_denseShelf_4070.jpg" width="100%" /></td>
    <td><img src="assets/cur_shelf_4070.jpg" width="100%" /></td>
    <td><img src="assets/cur_shelfSimple_4070.jpg" width="100%" /></td>
  </tr>
  <tr>
    <td align="center">Dense Scene</td>
    <td align="center">Medium Scene</td>
    <td align="center">Simple Scene</td>
  </tr>
</table>
<p align="center">
  <em>Our piecewise-linear and discretized methods outpaced cuRobo on all except large batch sizes on simple scenes, with discretization being the fastest on dense scenes at the cost of recall rate. Quadratic B-splines had the best accuracy at slower speeds, but were still faster than cuRobo at most batch sizes for medium/dense scenes.</em>
</p>

## Setup
### Hardware
A RTX GPU is required to run this program.

### Install Dependencies
I used vcpkg to manage most of the dependencies. You can install it from [here](https://vcpkg.io/en/).
After installing vcpkg, you can install the dependencies by running the following command:
```bash
vcpkg install eigen3 urdfdom urdfdom-headers lz4 benchmark tbb nlohmann-json FLANN gtest imgui glfw3
```

You also need to install CUDAToolKit (>=12.6), which you can download from [here](https://developer.nvidia.com/cuda-downloads).
OptiX is also required (>=7.7) and can be downloaded from [here](https://developer.nvidia.com/designworks/optix/download).

Set up the environment variable for CUDA and configure OptiX_ROOT_DIR.

### Build
You can build the project using CMake:
```bash
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=$PATH_TO_YOUR_VCPKG$/scripts/buildsystems/vcpkg.cmake
```

## Usage
This is a demo of the collision checker. You can run it by executing the following commands:
```bash
cd build
cmake --build . --target AllDemos --config Release
cd ./bin/Demos/Release   # If on Windows
cd ./bin/Demos           # If on Linux
# Run any executable here with a name starting with Demo
./demoQuadContinuous.exe       # For example
```
## Setup a new robot

Coming soon. Stay tune!

## Setup a new mesh collision scene

Coming soon. Stay tune!

## Benchmarks

### Collision Scene
<p align="center">
  <img src="assets/collisionScene.png" width="96%" />
</p>

<p align="center">
  <em>All benchmark is carried out in a scene like this. The simple, medium, and dense versions of the scene consisted of over 15k / 22k, 71k / 107k, and 191k / 321k triangles / edges, respectively.</em>
</p>

### RTCD Benchmarks

**Make sure you run this before you run the curobo benchmark. The RTCD benchmark will generate the poses used for benchmarking.**

```bash
cd build
cmake --build . --target AllBenchmarks --config Release
cd ..
./benchmark.bat     # If on Windows
./benchmark.sh      # If on Linux
```

This will take quite amount of time to run all benchmarks.

### Curobo Benchmarks
We modified the curobo library to disable self-collision and other constraints in order for a fair comparison. To run the curobo benchmark, first install a customized version of curobo [here](https://github.com/Ssz990220/curobo).

The scripts for curobo benchmarking can be found in `scripts/Benchmark/curobo`.

### FCL Benchmarks

We use Moveit Interface for FCL benchmarking. Benchmark setup and source code can be found [here](https://github.com/Ssz990220/rtcc_benchmark).

### Plot the result
**To plot the result, make sure you run RTCD Benchmarks and Curobo Benchmarks first.**

First you need to install julia (>1.10 recommended). And `cd scripts`
```julia
using Pkg
Pkg.activate(.)
# Hit "]", the terminal should start with `(scripts) pkg>`
instantiate # This would install all the required packages for plotting and may take a while.
```
With the above command, you setup the env. Now back to a bash to plot the results.
```bash
julia -O3 ./scripts/Benchmark/plots/discrete/resultAnalysis.jl # For discrete poses benchmark results
julia -O3 ./scripts/Benchmark/plots/curve/resultAnalysisEff.jl # For continuous benchmark results
```

The plots will be in `./data/Plots/

## Disclaimers
The code contained herein is of "research grade", meaning it is messy, largely uncommented, insufficently reviewed and optimzied, and intended for reference more than anything else.
The authors have tried to make it as "user-friendly" as possible. However, the best we can do now is make it compile and run.

Further development is ongoing. For upcoming features and demos, please refer to the TODO list.

## Acknowledgements
This project is based on the following projects:
- Official OptiX SDK samples shipped with OptiX
- [optix7course](https://github.com/ingowald/optix7course)

## Citing
This repository was developed at the [Human Centered Robotics Lab (HCRL)](https://sites.utexas.edu/hcrl/) of The University of Texas at Austin in collaboration with Dexterity, Inc. by [Sizhe Sui](https://ssz990220.github.io/), advised by [Prof. Luis Sentis](https://scholar.google.com/citations?user=-3pL5qkAAAAJ&hl=en) and [Andrew Bylard](https://scholar.google.com/citations?user=wKr1q1IAAAAJ&hl=en).

If you found this repository useful, please cite our paper:
- [1] **Conference Paper:** S Sui, L Sentis, and A Bylard, *Hardware-Accelerated Ray Tracing for Discrete and Continuous Collision Detection on GPUs.* **IEEE International Conference on Robotics and Automation (ICRA)** 2025. Available: [https://www.arxiv.org/abs/2409.09918](https://www.arxiv.org/abs/2409.09918)
