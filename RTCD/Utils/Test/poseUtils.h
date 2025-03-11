#pragma once
#include <Planner/KNN/knn.h>
#include <Planner/sampler/sampler.h>
#include <Utils/ArrayMath.h>
#include <config.h>
#include <math.h>

using namespace RTCD;
#ifdef _WIN32
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif
#endif

template <size_t Dim>
inline void randomMotions(const size_t nSample, std::vector<std::array<float, Dim>>& samples,
    std::array<float, Dim> upper, std::array<float, Dim> lower, std::string_view name) {
    // Check if the file already exsits.
    std::string filename = CONCAT_PATHS(PROJECT_BASE_DIR, "/data/motions/");
    // if not exists, create the folder
    std::filesystem::create_directories(filename);
    // Append the name and number of samples to the filename
    filename             = filename + std::string(name) + std::to_string(nSample) + ".bin";
    // if it exists, load the trajectory from the file
    if (std::filesystem::exists(filename)) {
        readBin(filename, samples);
#ifdef PRINTIT
        std::cout << "Loaded trajectory from file: " << filename << std::endl;
#endif
        // double check if the size match
        if (samples.size() != nSample) {
            std::cerr << "The size of the loaded trajectory " << samples.size() << " does not match the expected size "
                      << nSample << std::endl;
            exit(1);
        }
    } else {
        // if it does not exist, generate the trajectory
        createSamplesHalton<Dim>(nSample, samples, lower, upper);
        // create the folder if it does not exist
        std::filesystem::create_directory(CONCAT_PATHS(PROJECT_BASE_DIR, "/data/motions"));
        writeBin(filename, samples);
        std::cout << "Generated trajectory and saved to file: " << filename << std::endl;
    }
}

template <size_t n>
std::string concatFloatsAsInts(const std::array<float, n>& arr) {
    std::stringstream ss;
    for (const float& f : arr) {
        // Convert each float to an int and append to the stringstream
        ss << static_cast<int>(f);
    }
    return ss.str(); // Convert the stringstream to string and return
}

template <size_t dim, int nCSpacePts, int KNN_K>
void KNearestNeighbor(std::vector<std::array<float, dim>>& CSpacePts, std::vector<float>& distance,
    std::vector<int>& indices, std::string_view name, const std::array<float, dim>& weight) {

    std::string weightStr = concatFloatsAsInts(weight);
    // Check if the file already exsits.
    std::string filename = CONCAT_PATHS(PROJECT_BASE_DIR, "/data/motions/");
    std::string idxName  = filename + std::string(name) + "/" + weightStr + "/" + std::to_string(nCSpacePts) + "_knn"
                        + std::to_string(KNN_K) + "_idx.bin";
    std::string disName = filename + std::string(name) + "/" + weightStr + "/" + std::to_string(nCSpacePts) + "_knn"
                        + std::to_string(KNN_K) + "_dis.bin";
    // create the folder if it does not exist
    std::filesystem::create_directories(filename + std::string(name) + "/" + weightStr);
    // if it exists, load the trajectory from the file
    if (std::filesystem::exists(idxName) && std::filesystem::exists(disName)) {
        readBin(idxName, indices);
        readBin(disName, distance);
        // std::cout << "Loaded KNN from file: " << idxName << std::endl;
        // double check if the size match
        if (indices.size() != nCSpacePts * KNN_K || distance.size() != nCSpacePts * nCSpacePts) {
            std::cerr << "The size of the loaded KNN result does not match the expected size" << std::endl;
            exit(1);
        }
    } else {
        // make the traj row major for KNN
        std::vector<float> hostPosesRowMajor(dim * nCSpacePts);
        for (int i = 0; i < nCSpacePts; i++) {
            for (int j = 0; j < dim; j++) {
                hostPosesRowMajor[j * nCSpacePts + i] = CSpacePts[i][j];
            }
        }

        // KNN Setup
        KNN::KNN<nCSpacePts, dim, KNN_K> knn;
        knn.setWeight(weight.data());
        knn.setRef(hostPosesRowMajor.data(), 0, cudaMemcpyHostToDevice);
        CUDA_SYNC_CHECK();
        CUDABuffer mask; // Assume all poses are valid
        mask.alloc(nCSpacePts * sizeof(float));
        knn.NN(0);
        CUDA_SYNC_CHECK();

        distance.resize(nCSpacePts * nCSpacePts);
        indices.resize(nCSpacePts * KNN_K);
        knn.downloadResult(distance.data(), indices.data(), 0, cudaMemcpyDeviceToHost);
        CUDA_SYNC_CHECK();

        // Write the result
        writeBin(idxName, indices);
        writeBin(disName, distance);
        std::cout << "Generated KNN and saved to file: " << idxName << " and " << disName << std::endl;
    }
}

// genCSpaceKNNTraj()
// - nCSpacePts: number of sample points in the configuration space
// - nTrajPts: number of points in one trajectory (inclusive)
// - upperBound: upper bound of the configuration space
// - lowerBound: lower bound of the configuration space
template <size_t dim, int nCSpacePts, int KNN_K>
void genCSpaceKNNTraj(std::vector<std::array<float, dim>>& traj, int nTrajPts, const std::array<float, dim>& upperBound,
    const std::array<float, dim>& lowerBound, std::string_view robotName, const std::array<float, dim>& weight) {
    // Prepare CSpace Sample Points
    std::vector<std::array<float, dim>> CSpacePts(nCSpacePts);
    randomMotions(nCSpacePts, CSpacePts, upperBound, lowerBound, robotName);

    // Compute KNN, so that the length of the traj is controlled by the number of CSpace samples
    std::vector<float> distances(nCSpacePts * nCSpacePts);
    std::vector<int> indices(nCSpacePts * KNN_K);
    KNearestNeighbor<dim, nCSpacePts, KNN_K>(CSpacePts, distances, indices, robotName, weight);

    // generate the trajectory based on the indices
    std::array<float, dim> start;
    std::array<float, dim> end;
    std::array<float, dim> span;
    for (int i = 0; i < nCSpacePts; i++) {
        start = CSpacePts[i];
        for (int j = 0; j < KNN_K; j++) {
            end  = CSpacePts[indices[i * KNN_K + j]];
            span = (end - start) / (float) (nTrajPts - 1);
            for (int k = 0; k < nTrajPts; k++) {
                traj[i * KNN_K * nTrajPts + j * nTrajPts + k] = start + span * (float) k;
            }
        }
    }
}


///////////////////////////////////////////////////////////////
//                       Deprecated Code                     //
//        Not deleted because it is used in the tests        //
///////////////////////////////////////////////////////////////


template <size_t Dim>
inline void genTraj(
    const size_t trajLength, std::array<float, Dim>& start, std::vector<std::array<float, Dim>>& traj, float dt) {};

template <>
inline void genTraj<6>(
    const size_t trajLength, std::array<float, 6>& start, std::vector<std::array<float, 6>>& traj, float dt) {
    for (size_t i = 0; i < trajLength; ++i) {
        float t    = i * dt;
        traj[i][0] = start[0] + 0.8 * sinf(2.0 * t);
        traj[i][1] = start[1] + 1.2f * cosf(0.3 * t + 0.2 * M_PI);
        traj[i][2] = start[2] + 0.3 * cosf(-0.2 * t - 0.2 * M_PI);
        traj[i][3] = start[3] + 1.2 * sinf(0.2 * t + 0.2 * M_PI);
        traj[i][4] = start[4] + 0.8 * sinf(1.3 * t - 0.5 * M_PI);
        traj[i][5] = start[5] + 0.2 * sinf(0.3 * t);
    }
}

template <>
inline void genTraj<7>(
    const size_t trajLength, std::array<float, 7>& start, std::vector<std::array<float, 7>>& traj, float dt) {
    for (size_t i = 0; i < trajLength; ++i) {
        float t    = i * dt;
        traj[i][0] = start[0] + 0.8 * sinf(2.0 * t);
        traj[i][1] = start[1] + 1.2f * cosf(0.3 * t + 0.2 * M_PI);
        traj[i][2] = start[2] + 0.3 * cosf(-0.2 * t - 0.2 * M_PI);
        traj[i][3] = start[3] + 1.2 * sinf(0.2 * t + 0.2 * M_PI);
        traj[i][4] = start[4] + 0.8 * sinf(1.3 * t - 0.5 * M_PI);
        traj[i][5] = start[5] + 0.2 * sinf(0.3 * t);
        traj[i][6] = start[6] + 0.3 * cosf(0.4 * t);
    }
}

template <size_t Dim>
inline void randomTrajs(const size_t nTraj, const size_t trajLength, std::vector<std::array<float, Dim>>& traj,
    std::array<float, Dim>& upper, std::array<float, Dim>& lower, float dt = 0.01f) {
    traj.clear();
    std::vector<std::array<float, Dim>> starts(nTraj);
    createSamplesHalton<Dim>(nTraj, starts, lower, upper);
    for (int i = 0; i < nTraj; ++i) {
        std::vector<std::array<float, Dim>> trajLocal(trajLength);
        genTraj<Dim>(trajLength, starts[i], trajLocal, dt);
        traj.insert(traj.end(), trajLocal.begin(), trajLocal.end());
    }
}


template <size_t Dim>
void importBenchmarkTraj(const size_t nTrajPts, std::vector<std::array<float, Dim>>& traj) {}

// TODO: implement the following function
template <>
void importBenchmarkTraj(const size_t nTrajPts, std::vector<std::array<float, 6>>& traj) {}

template <>
void importBenchmarkTraj(const size_t nTrajPts, std::vector<std::array<float, 7>>& traj) {
    // First load the trajectory from file

    std::string trajFile = CONCAT_PATHS(PROJECT_BASE_DIR, "/simpleCollisionChecker/data/benchmark/trajPosesPanda.bin");
    std::ifstream file(trajFile, std::ios::binary);
    std::vector<std::array<float, 7>> poses;
    if (file.is_open()) {
        while (true) {
            std::array<float, 7> posef;
            file.read(reinterpret_cast<char*>(posef.data()), sizeof(float) * 7);
            if (file.eof()) {
                break;
            }
            poses.push_back(posef);
        }
    } else {
        std::cerr << "Failed to open file: " << trajFile << std::endl;
        exit(1);
    }
    file.close();

    // Then interpolate the trajectory to have nTrajPts points
    const size_t nTraj = poses.size() / 2;
    traj.clear();
    for (size_t i = 0; i < nTraj; ++i) {
        std::array<float, 7> start = poses[i * 2];
        std::array<float, 7> end   = poses[i * 2 + 1];
        std::array<float, 7> span  = end - start;
        for (size_t j = 0; j < nTrajPts; ++j) {
            std::array<float, 7> pose = start + span * static_cast<float>(j) / static_cast<float>(nTrajPts - 1);
            traj.push_back(pose);
        }
    }
}


template <size_t Dim>
inline void motionGen(float t, std::array<float, Dim>& q) {
    // q[2] = q[3] = q[4] = q[5] = q[6] = 0.f;
}

template <>
inline void motionGen<6>(float t, std::array<float, 6>& q) {
    q[0] = 0.8 * sinf(2.0 * t);
    // q[0] = 0.f;
    q[1] = 1.2f * cosf(0.3 * t + 0.2 * M_PI);
    q[2] = 0.3 * cosf(-0.2 * t - 0.2 * M_PI);
    q[3] = 1.2 * sinf(0.2 * t + 0.2 * M_PI);
    q[4] = 0.8 * sinf(1.3 * t - 0.5 * M_PI);
    q[5] = 0.2 * sinf(0.3 * t);
}

template <>
inline void motionGen<7>(float t, std::array<float, 7>& q) {
    q[0] = 0.8 * sinf(2.0 * t);
    // q[0] = 0.f;
    q[1] = 1.2f * cosf(0.3 * t + 0.2 * M_PI);
    q[2] = 0.3 * cosf(-0.2 * t - 0.2 * M_PI);
    q[3] = 1.2 * sinf(0.2 * t + 0.2 * M_PI);
    q[4] = 0.8 * sinf(1.3 * t - 0.5 * M_PI);
    q[5] = 0.2 * sinf(0.3 * t);
    q[6] = 0.3 * cosf(0.4 * t);
}

template <>
inline void motionGen<8>(float t, std::array<float, 8>& q) {
    q[0] = 0.8 * sinf(2.0 * t);
    // q[0] = 0.f;
    q[1] = 1.2f * cosf(0.3 * t + 0.2 * M_PI);
    q[2] = 0.3 * cosf(-0.2 * t - 0.2 * M_PI);
    q[3] = 1.2 * sinf(0.2 * t + 0.2 * M_PI);
    q[4] = 0.8 * sinf(1.3 * t - 0.5 * M_PI);
    q[5] = 0.2 * sinf(0.3 * t);
    q[6] = 0.3 * cosf(0.4 * t);
    q[7] = 0.3 * cosf(0.4 * t);
}
