// rtcd_core: the one definition of the OptiX function table (OPTIX_FUNCTION_TABLE_SYMBOL) that optix_stubs.h needs.
//
// It is deliberately ALONE in this object file: a static library member is only linked when it resolves an
// undefined symbol, so an executable that already defines the table itself (old-stack code that includes
// RTCD/Utils/Test/cudaCtxUtils.h -> optix_function_table_definition.h, e.g. parity tests linking the old stack)
// does not pull this object in and gets no duplicate definition. Do not add anything else to this file.
#include <optix_function_table_definition.h>
