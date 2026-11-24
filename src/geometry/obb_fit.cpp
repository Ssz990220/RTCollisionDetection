// fitObb: the old RTCD::computeBestFittingOBB (RTCD/Meshes/Obb.h), statement by statement with the same Eigen
// types and expressions so the float results are identical (timing printout dropped; an empty set throws instead of
// returning a zero OBB).

#include <rtcd/errors.h>
#include <rtcd/geometry.h>

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <limits>
#include <utility>

namespace rtcd {

    namespace {
        Eigen::Vector3f toEigen(const Vec3& v) {
            return Eigen::Vector3f(v[0], v[1], v[2]);
        }
        void store(float (&dst)[3], const Eigen::Vector3f& v) {
            dst[0] = v.x();
            dst[1] = v.y();
            dst[2] = v.z();
        }
    } // namespace

    Obb fitObb(std::span<const Vec3> points) {
        if (points.empty()) {
            throw InvalidArgument("fitObb: empty point set");
        }

        // 1. mean
        Eigen::Vector3f mean = Eigen::Vector3f::Zero();
        for (const auto& p : points) {
            mean += toEigen(p);
        }
        mean /= static_cast<float>(points.size());

        // 2. covariance (1/N) sum (p - mean)(p - mean)^T
        Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
        for (const auto& p : points) {
            Eigen::Vector3f diff = toEigen(p) - mean;
            covariance += diff * diff.transpose();
        }
        covariance /= static_cast<float>(points.size());

        // 3. eigen decomposition
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigenSolver(covariance);
        Eigen::Vector3f eigenValues  = eigenSolver.eigenvalues();
        Eigen::Matrix3f eigenVectors = eigenSolver.eigenvectors();

        // 4. eigenvalues descending
        std::array<std::pair<float, int>, 3> eigenValIdx;
        for (int i = 0; i < 3; ++i) {
            eigenValIdx[static_cast<std::size_t>(i)] = std::make_pair(eigenValues[i], i);
        }
        std::sort(eigenValIdx.begin(), eigenValIdx.end(), [](auto& a, auto& b) { return a.first > b.first; });

        Eigen::Matrix3f R_sorted;
        for (int i = 0; i < 3; ++i) {
            R_sorted.col(i) = eigenVectors.col(eigenValIdx[static_cast<std::size_t>(i)].second);
        }

        // 5. right-handed
        if (R_sorted.determinant() < 0.f) {
            R_sorted.col(2) *= -1.f;
        }

        // 6. extents in the principal frame
        Eigen::Vector3f minProj = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
        Eigen::Vector3f maxProj = Eigen::Vector3f::Constant(-std::numeric_limits<float>::max());
        for (const auto& p : points) {
            Eigen::Vector3f pLocal = R_sorted.transpose() * (toEigen(p) - mean);
            minProj                = minProj.cwiseMin(pLocal);
            maxProj                = maxProj.cwiseMax(pLocal);
        }

        // 7. / 8. half extents, center
        Eigen::Vector3f halfExtents = 0.5f * (maxProj - minProj);
        Eigen::Vector3f localCenter = 0.5f * (maxProj + minProj);
        Eigen::Vector3f worldCenter = mean + R_sorted * localCenter;

        // 9. fill
        Obb obb{};
        store(obb.center, worldCenter);
        store(obb.halfSize, halfExtents);
        for (int i = 0; i < 3; ++i) {
            Eigen::Vector3f axis = R_sorted.col(i).normalized();
            store(obb.axes[i], axis);
        }
        return obb;
    }

} // namespace rtcd
