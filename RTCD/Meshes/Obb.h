#pragma once

#include <Eigen/Dense>
#include <Meshes/meshTypes.h>
#include <chrono>
#include <limits>
#include <numeric>
#include <vector>
#undef max

namespace RTCD {
    // Convert float3 <-> Eigen::Vector3f
    inline Eigen::Vector3f toEigen(const float3& f3) {
        return Eigen::Vector3f(f3.x, f3.y, f3.z);
    }

    inline float3 make_float3(float x, float y, float z) {
        float3 r;
        r.x = x;
        r.y = y;
        r.z = z;
        return r;
    }

    inline float3 fromEigen(const Eigen::Vector3f& v) {
        return make_float3(v.x(), v.y(), v.z());
    }

    OBB computeBestFittingOBB(const std::vector<float3>& points) {
        OBB obb;

        auto start = std::chrono::high_resolution_clock::now();
        // 0. Edge case: if no points, just return default OBB
        if (points.empty()) {
            return obb;
        }

        // 1. Compute the mean (centroid) of the points
        Eigen::Vector3f mean = Eigen::Vector3f::Zero();
        for (const auto& p : points) {
            mean += toEigen(p);
        }
        mean /= static_cast<float>(points.size());

        // 2. Compute the covariance matrix
        //    C = (1 / N) * Σ( (p - mean) * (p - mean)^T )
        Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
        for (const auto& p : points) {
            Eigen::Vector3f diff = toEigen(p) - mean;
            covariance += diff * diff.transpose();
        }
        covariance /= static_cast<float>(points.size());

        // 3. Perform eigen-decomposition on the covariance matrix
        //    Eigenvectors => principal axes, Eigenvalues => variance along those axes
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigenSolver(covariance);
        Eigen::Vector3f eigenValues  = eigenSolver.eigenvalues();
        Eigen::Matrix3f eigenVectors = eigenSolver.eigenvectors();

        // 4. The solver returns eigenvalues in ascending order. We want them descending.
        //    Build array of (eigenvalue, index), then sort by eigenvalue descending.
        std::array<std::pair<float, int>, 3> eigenValIdx;
        for (int i = 0; i < 3; ++i) {
            eigenValIdx[i] = std::make_pair(eigenValues[i], i);
        }
        std::sort(eigenValIdx.begin(), eigenValIdx.end(), [](auto& a, auto& b) { return a.first > b.first; });

        // Reorder eigenvalues and eigenvectors
        Eigen::Matrix3f R_sorted;
        for (int i = 0; i < 3; ++i) {
            R_sorted.col(i) = eigenVectors.col(eigenValIdx[i].second);
        }

        // 5. Ensure right-handed coordinate system (optional).
        //    If the determinant is negative, flip one axis.
        if (R_sorted.determinant() < 0.f) {
            R_sorted.col(2) *= -1.f;
        }

        // 6. Transform points into principal-axis space to find min/max extents
        Eigen::Vector3f minProj = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
        Eigen::Vector3f maxProj = Eigen::Vector3f::Constant(-std::numeric_limits<float>::max());

        for (const auto& p : points) {
            Eigen::Vector3f pLocal = R_sorted.transpose() * (toEigen(p) - mean);
            minProj                = minProj.cwiseMin(pLocal);
            maxProj                = maxProj.cwiseMax(pLocal);
        }

        // 7. Compute half extents and center in local space
        Eigen::Vector3f halfExtents = 0.5f * (maxProj - minProj);
        Eigen::Vector3f localCenter = 0.5f * (maxProj + minProj);

        // 8. Convert the local center back to world space
        Eigen::Vector3f worldCenter = mean + R_sorted * localCenter;

        // 9. Fill the OBB structure
        obb.center   = fromEigen(worldCenter);
        obb.halfSize = fromEigen(halfExtents);

        // Fill the orientation vectors (principal axes):
        // R_sorted.col(0) is the X-axis, col(1) is the Y-axis, col(2) is the Z-axis
        for (int i = 0; i < 3; ++i) {
            Eigen::Vector3f axis = R_sorted.col(i).normalized(); // Should already be normalized
            obb.orientation[i]   = fromEigen(axis);
        }

        auto end                              = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Time to compute OBB: " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count()
                  << "us for a mesh with " << points.size() << " vertices." << std::endl;

        return obb;
    }


} // namespace RTCD
