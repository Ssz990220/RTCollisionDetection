#pragma once

#include <config.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

template <typename T>
void readBin(const std::string& filename, std::vector<T>& data) {
    std::fstream file(filename, std::ios::in | std::ios::binary);
    data.clear();
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(1);
    } else {
        while (true) {
            T t;
            file.read(reinterpret_cast<char*>(&t), sizeof(T));
            if (file.eof()) {
                break;
            }
            data.push_back(t);
        }
    }
    file.close();
}

template <typename T, size_t Dim>
void readBin(const std::string& filename, std::vector<std::array<T, Dim>>& data) {
    std::fstream file(filename, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(1);
    } else {
        data.clear();
        while (true) {
            std::array<T, Dim> t;
            file.read(reinterpret_cast<char*>(t.data()), sizeof(T) * Dim);
            if (file.eof()) {
                break;
            }
            data.push_back(t);
        }
    }
    file.close();
}

template <typename T>
void writeBin(const std::string& filename, const std::vector<T>& data) {
    // get dir name from filename
    std::filesystem::path dir = std::filesystem::path(filename).parent_path();
    if (!std::filesystem::exists(dir)) {
        std::filesystem::create_directories(dir);
    }
    std::fstream file(filename, std::ios::out | std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(1);
    } else {
        for (const auto& d : data) {
            file.write(reinterpret_cast<const char*>(&d), sizeof(T));
        }
    }
    file.close();
}

template <typename T, size_t Dim>
void writeBin(const std::string& filename, const std::vector<std::array<T, Dim>>& data) {
    std::fstream file(filename, std::ios::out | std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(1);
    } else {
        for (const auto& d : data) {
            file.write(reinterpret_cast<const char*>(d.data()), sizeof(T) * Dim);
        }
    }
    file.close();
}


template <size_t Dim>
void loadTraj(std::vector<std::array<float, Dim>>& traj, std::string_view robotName, std::string_view weightStr,
    int nSample, int KNN_K, int trajId, int nTPts, std::vector<int>& poseInTraj) {
    std::string_view motionDir = CONCAT_PATHS(PROJECT_BASE_DIR, "/data/motions/");
    std::string trajFile = std::string(motionDir) + "/" + std::string(robotName) + "/" + std::string(weightStr) + "/"
                         + std::to_string(nSample) + "_knn" + std::to_string(KNN_K) + "_" + std::to_string(nTPts)
                         + "_traj.bin";
    std::cout << "Loading trajectory from: " << trajFile << std::endl;
    if (!std::filesystem::exists(trajFile)) {
        std::cerr << "Trajectory file not found: " << trajFile << std::endl;
        exit(1);
    }
    std::vector<std::array<float, Dim>> trajData;
    readBin(trajFile, trajData);
    if (trajId * nTPts >= trajData.size()) {
        std::cerr << "Trajectory ID out of bound: " << trajId << " >= " << trajData.size() << std::endl;
        exit(1);
    } else {
        // Copy the trajectory to the traj vector
        traj.clear();
        for (auto pose : poseInTraj) {
            std::array<float, Dim> tmp = trajData.at(trajId * nTPts + pose);
            traj.push_back(trajData.at(trajId * nTPts + pose));
        }
    }
}
