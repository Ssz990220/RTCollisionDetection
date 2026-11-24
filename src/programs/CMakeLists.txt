# rtcd_programs: OptiX IR device programs of the new stack (raygen / any-hit / miss per backend), compiled to
# OptiX IR and embedded into this static library (see rtcd_add_optix_ir in cmake/RTCDTargets.cmake for the symbol
# names and the generated header <rtcd_programs_embedded.h>). Add one .cu per program module to SOURCES.
rtcd_add_optix_ir(rtcd_programs
  SOURCES placeholder.cu
)
