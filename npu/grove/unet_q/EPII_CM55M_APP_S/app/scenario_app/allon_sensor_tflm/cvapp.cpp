#include <cstdio>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "WE2_device.h"
#include "board.h"
#include "cvapp.h"

#include "WE2_core.h"
#include "WE2_device.h"

#include "ethosu_driver.h"
#include "core_cm55.h"  // CMSIS header for Cortex-M55
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

#include "xprintf.h"

#include "yolo.h"
#include "common_config.h"

#define LOCAL_FRAQ_BITS (8)
#define SC(A, B) ((A<<8)/B)

#define INPUT_SIZE_X 96
#define INPUT_SIZE_Y 96

#ifdef TRUSTZONE_SEC
#define U55_BASE	BASE_ADDR_APB_U55_CTRL_ALIAS
#else
#ifndef TRUSTZONE
#define U55_BASE	BASE_ADDR_APB_U55_CTRL_ALIAS
#else
#define U55_BASE	BASE_ADDR_APB_U55_CTRL
#endif
#endif

#define CORE_CLOCK_HZ 100000000 // TODO get as argument from app_main
#define MODEL_IN_W	224 // TODO get as argument from app_main
#define MODEL_IN_H  224 // TODO get as argument from app_main
#define MODEL_IN_C	3 // TODO get as argument from app_main

#define TENSOR_ARENA_BUFSIZE  (512000)
__attribute__(( section(".bss.NoInit"))) uint8_t tensor_arena_buf[TENSOR_ARENA_BUFSIZE] __ALIGNED(32);

void enable_dwt_cycle_counter();
uint32_t get_time_cycles();
uint32_t cycles_to_us(uint32_t cycles, uint32_t core_clock_hz);

using namespace std;

namespace 
{
constexpr int tensor_arena_size = TENSOR_ARENA_BUFSIZE;
uint32_t tensor_arena = (uint32_t)tensor_arena_buf;

struct ethosu_driver ethosu_drv; /* default Ethos-U device driver */
tflite::MicroInterpreter *int_ptr=nullptr;
TfLiteTensor* input, *output;
};

void enable_dwt_cycle_counter() {
    // enable DWT and its cycle counter
    if (!(CoreDebug->DEMCR & CoreDebug_DEMCR_TRCENA_Msk)) {
        CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk; 
    }
    DWT->CYCCNT = 0;               // reset counter
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;  // enable counter
}

uint32_t get_time_cycles() {
    return DWT->CYCCNT;
}

uint32_t cycles_to_us(uint32_t cycles, uint32_t core_clock_hz) {
    return cycles / (core_clock_hz / 1000000);
}

static void generate_random_image(uint8_t* image) 
{
    // for (int i = 0; i < width * height; ++i) 
    // {
    //     image[i] = (rand() % 256) - 128; // generate random pixel value between -128 to 127
    // }
    for (int i = 0; i < MODEL_IN_W; i++)
    {
        for (int j = 0; j < MODEL_IN_H; j++)
        {
            for (int c = 0; c < MODEL_IN_C; c++)
            {
                *image++ = 0;
            }
        }
    }
}

static void _arm_npu_irq_handler(void)
{
    /* calls default interrupt handler from NPU driver */
    ethosu_irq_handler(&ethosu_drv);
}

/**
 * @brief  initalises NPU IRQ
 **/
static void _arm_npu_irq_init(void)
{
    const IRQn_Type ethosu_irqnum = (IRQn_Type)U55_IRQn;

    /* registers Ethos-U IRQ handler in our vector table
     * note: handler comes from Ethos-U driver */
    EPII_NVIC_SetVector(ethosu_irqnum, (uint32_t)_arm_npu_irq_handler);

    /* enables IRQ */
    NVIC_EnableIRQ(ethosu_irqnum);

}

static int _arm_npu_init(bool security_enable, bool privilege_enable)
{
    int err = 0;

    /* init IRQ */
    _arm_npu_irq_init();

    /* init Ethos-U55 device */
    const void * ethosu_base_address = (void *)(U55_BASE);

    if (0 != (err = ethosu_init(
                            &ethosu_drv,             /* Ethos-U driver device pointer */
                            ethosu_base_address,     /* Ethos-U NPU's base address. */
                            NULL,       /* pointer to fast mem area - NULL for U55. */
                            0, /* fast mem region size. */
						security_enable,                       /* security enable. */
						privilege_enable))) {                  /* privilege enable. */
    	xprintf("failed to initalise Ethos-U device\n");
            return err;
        }

    xprintf("Ethos-U55 device initialised\n");

    return 0;
}

int cv_init(bool security_enable, bool privilege_enable)
{
	int ercode = 0;
    uint32_t start_cycles, end_cycles, elapsed_cycles;

    start_cycles = get_time_cycles();
	if(_arm_npu_init(security_enable, privilege_enable)!=0)
		return -1;
    end_cycles = get_time_cycles();
    printf("npu init: %lu us\n", cycles_to_us(end_cycles - start_cycles, CORE_CLOCK_HZ));

    start_cycles = get_time_cycles();

#if (FLASH_XIP_MODEL == 1)
	static const tflite::Model*model = tflite::GetModel((const void *)0x3A180000);
#else
	static const tflite::Model*model = tflite::GetModel((const void *)g_model_data_vela);
#endif

	if (model->version() != TFLITE_SCHEMA_VERSION) 
    {
		xprintf(
			"[ERROR] model's schema version %d is not equal "
			"to supported version %d\n",
			model->version(), TFLITE_SCHEMA_VERSION);
		return -1;
	}
	else 
    {
		xprintf("model's schema version %d\n", model->version());
	}

	static tflite::MicroErrorReporter micro_error_reporter;
	static tflite::MicroMutableOpResolver<8> op_resolver;

	if (kTfLiteOk != op_resolver.AddEthosU())
    {
		xprintf("failed to add Arm NPU support to op resolver.");
		return false;
	}
    // op_resolver.AddPad();
    // op_resolver.AddConcatenation();
    // op_resolver.AddSlice();
    op_resolver.AddQuantize();
    // op_resolver.AddResizeNearestNeighbor();
    // op_resolver.AddSplit();
    op_resolver.AddConv2D();
    op_resolver.AddDequantize();
    op_resolver.AddFloor();
    // op_resolver.AddRelu();
    // op_resolver.AddMul();
    // op_resolver.AddAdd();
    op_resolver.AddMinimum();
    op_resolver.AddMaximum();
    // op_resolver.AddMaxPool2D();
    // op_resolver.AddSoftmax();
    // op_resolver.AddReshape();
    op_resolver.AddTranspose(); // TODO remove https://github.com/HimaxWiseEyePlus/YOLOv8_on_WE2 

    printf("here\n");
	static tflite::MicroInterpreter static_interpreter(model, op_resolver, (uint8_t*)tensor_arena, tensor_arena_size, &micro_error_reporter);
    printf("now here\n");

	if(static_interpreter.AllocateTensors()!= kTfLiteOk) 
    {
		return false;
	}
    printf("and now here\n");

	int_ptr = &static_interpreter;
	input = static_interpreter.input(0);
	output = static_interpreter.output(0);

    end_cycles = get_time_cycles();
    printf("model init: %lu us\n", cycles_to_us(end_cycles - start_cycles, CORE_CLOCK_HZ));

	return ercode;
}

int cv_run() {
    enable_dwt_cycle_counter();  // Enable the DWT cycle counter
	int ercode = 0;

	uint8_t random_image[MODEL_IN_C * MODEL_IN_H * MODEL_IN_W];
	generate_random_image(random_image);

	// loading input?
    uint32_t start_cycles, end_cycles, elapsed_cycles;
    start_cycles = get_time_cycles();
    for (int i = 0; i < INPUT_SIZE_X * INPUT_SIZE_Y; i++) 
    {
        input->data.int8[i] = random_image[i];
    }
    end_cycles = get_time_cycles();
    printf("memory load: %lu us\n", cycles_to_us(end_cycles - start_cycles, CORE_CLOCK_HZ));

    // inference
    start_cycles = get_time_cycles();
    TfLiteStatus invoke_status = int_ptr->Invoke();
    end_cycles = get_time_cycles();
    printf("inference: %lu us\n", cycles_to_us(end_cycles - start_cycles, CORE_CLOCK_HZ));

    if (invoke_status != kTfLiteOk) {
        printf("invoke failed\n");
        return -1;
    }

	// unloading output?
    start_cycles = get_time_cycles();
    int8_t score = output->data.int8[1];
    int8_t no_score = output->data.int8[0];
    end_cycles = get_time_cycles();
    printf("memory unload: %lu us\n", cycles_to_us(end_cycles - start_cycles, CORE_CLOCK_HZ));

    // CPU-processing e.g. softmax

	return ercode;
}

int cv_deinit()
{
	//TODO: add more deinit items here if need.
	return 0;
}
