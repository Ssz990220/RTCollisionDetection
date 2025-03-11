#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
#define HOSTDEVICE  __host__ __device__
#define RTCD_INLINE __forceinline__
#else
#define HOSTDEVICE
#define RTCD_INLINE inline
#endif
