#include <Utils/Test/testUtils.h>

int main() {
    std::vector<std::array<float, 7>> traj;

    importBenchmarkTraj(4, traj);
    std::cout << "Imported " << traj.size() << " points" << std::endl;
    return 0;
}
