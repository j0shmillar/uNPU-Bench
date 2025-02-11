#ifdef __arm__
#include <cmsis_compiler.h>
#else
#define __ALIGNED(x) __attribute__((aligned(x)))
#endif

#define MODEL_NAME "unet_large_q"
#define MODEL_INPUT_MEAN 127.5f
#define MODEL_INPUT_STD 127.5f