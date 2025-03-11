#include "sampler.h"
#include <iostream>

int main(){
    //std::vector<float> samples(NUM*DIM);
    RTCD::CUDABuffer samples;
    samples.allocManaged<float>(NUM * DIM);
    std::array<float,DIM> initial = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::array<float,DIM> goal = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    std::array<float,DIM> lo = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::array<float,DIM> hi = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    createSamplesHalton(0, (float*)samples.d_pointer(), initial, goal, lo, hi);
    for (int i = 0; i < NUM; ++i) {
        std::cout << ((float*)samples.d_pointer())[i*DIM] << " " << ((float*)samples.d_pointer())[i*DIM+1] << " " << ((float*)samples.d_pointer())[i*DIM+2] << " " << ((float*)samples.d_pointer())[i*DIM+3] << " " 
            << ((float*)samples.d_pointer())[i*DIM+4] << " " << ((float*)samples.d_pointer())[i*DIM+5] << " " << ((float*)samples.d_pointer())[i*DIM + 6] << std::endl;
    }
    return 0;
}