#include <cstdio>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <random>

#include "board.h"
#include "cvapp.h"

#include "WE2_core.h"
#include "WE2_device.h"

#include "ethosu_driver.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "xprintf.h"

#include "model_data.h"
#include "common_config.h"

#define LOCAL_FRAQ_BITS (8)
#define SC(A, B) ((A<<8)/B)

#define INPUT_SIZE_X 256
#define INPUT_SIZE_Y 3
#define INPUT_CHANNELS 1
#define OUTPUT_ROWS 3
#define OUTPUT_COLS 256
#define CHAR_BUFF_SIZE 50 
#define CPU_FREQ_MHZ 100

#ifdef TRUSTZONE_SEC
#define U55_BASE	BASE_ADDR_APB_U55_CTRL_ALIAS
#else
#ifndef TRUSTZONE
#define U55_BASE	BASE_ADDR_APB_U55_CTRL_ALIAS
#else
#define U55_BASE	BASE_ADDR_APB_U55_CTRL
#endif
#endif

#define INIT_DWT()      (DWT->CTRL |= 1) 
#define RESET_DWT()     (DWT->CYCCNT = 0) 
#define GET_DWT()       (DWT->CYCCNT) 

#define TENSOR_ARENA_BUFSIZE  (125*1024)
__attribute__(( section(".bss.NoInit"))) uint8_t tensor_arena_buf[TENSOR_ARENA_BUFSIZE] __ALIGNED(32);

static uint8_t random_image[INPUT_SIZE_X * INPUT_SIZE_Y * INPUT_CHANNELS];
static float processed_output[OUTPUT_ROWS * OUTPUT_COLS];

using namespace std;

namespace {

constexpr int tensor_arena_size = TENSOR_ARENA_BUFSIZE;
uint32_t tensor_arena = (uint32_t)tensor_arena_buf;

struct ethosu_driver ethosu_drv; 
tflite::MicroInterpreter *int_ptr=nullptr;
TfLiteTensor* input, *output;
};

static char * _float_to_char(float x, char *p) {
    char *s = p + CHAR_BUFF_SIZE - 1;  
    *s = '\0';  

    uint16_t decimals;
    int units;

    if (x < 0) { 
        decimals = (int)(x * -100) % 100; 
        units = (int)(-1 * x);
    } else { 
        decimals = (int)(x * 100) % 100;
        units = (int)x;
    }

    *--s = (decimals % 10) + '0';
    decimals /= 10; 
    *--s = (decimals % 10) + '0';
    *--s = '.';

    if (units == 0) {
        *--s = '0';
    } else {
        while (units > 0) {
            *--s = (units % 10) + '0';
            units /= 10;
        }
    }

    if (x < 0) *--s = '-';  

    return s;
}

void enable_dwt()
{
    if (!(CoreDebug->DEMCR & CoreDebug_DEMCR_TRCENA_Msk)) {
        CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk; 
        DWT->CYCCNT = 0; 
        DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk; 
    }
}

static void _arm_npu_irq_handler(void)
{
    ethosu_irq_handler(&ethosu_drv);
}

static void _arm_npu_irq_init(void)
{
    const IRQn_Type ethosu_irqnum = (IRQn_Type)U55_IRQn;
    EPII_NVIC_SetVector(ethosu_irqnum, (uint32_t)_arm_npu_irq_handler);
    NVIC_EnableIRQ(ethosu_irqnum);

}

static int _arm_npu_init(bool security_enable, bool privilege_enable)
{
    int err = 0;

    _arm_npu_irq_init();

#if TFLM2209_U55TAG2205
	const void * ethosu_base_address = (void *)(U55_BASE);
#else 
	void * const ethosu_base_address = (void *)(U55_BASE);
#endif
    if (0 != (err = ethosu_init(
                            &ethosu_drv,          
                            ethosu_base_address,    
                            NULL,    
                            0, 
							security_enable,       
							privilege_enable))) {            
    	xprintf("failed to initalise Ethos-U device\n");
            return err;
        }

    xprintf("Ethos-U55 device initialised\n");

    return 0;
}

void generate_random_image() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 255);
    
    for (int i = 0; i < INPUT_SIZE_X * INPUT_SIZE_Y * INPUT_CHANNELS; i++) {
        random_image[i] = static_cast<uint8_t>(dis(gen));
    }
}

void img_rescale_rgb(const uint8_t* in_image, int8_t* out_image) {
    for (int i = 0; i < INPUT_SIZE_X * INPUT_SIZE_Y * INPUT_CHANNELS; i++) {
        out_image[i] = static_cast<int8_t>(in_image[i] - 128);
    }
}

int cv_init(bool security_enable, bool privilege_enable)
{
	int ercode = 0;

	if(_arm_npu_init(security_enable, privilege_enable)!=0)
		return -1;

	static const tflite::Model*model = tflite::GetModel((const void *)g_model_data);

	if (model->version() != TFLITE_SCHEMA_VERSION) {
		xprintf(
			"[ERROR] model's schema version %d is not equal "
			"to supported version %d\n",
			model->version(), TFLITE_SCHEMA_VERSION);
		return -1;
	}
	else {
		xprintf("model's schema version %d\n", model->version());
	}

	static tflite::MicroMutableOpResolver<7> op_resolver;

    op_resolver.AddDepthwiseConv2D();
	op_resolver.AddRelu6();
	op_resolver.AddConv2D();
	op_resolver.AddAveragePool2D();
	op_resolver.AddReshape();
	op_resolver.AddSoftmax();
	if (kTfLiteOk != op_resolver.AddEthosU()){
		xprintf("failed to add Arm NPU support to op resolver.");
		return false;
	}

	static tflite::MicroInterpreter static_interpreter(model, op_resolver, (uint8_t*)tensor_arena, tensor_arena_size);

	if(static_interpreter.AllocateTensors()!= kTfLiteOk) {
		return false;
	}
	int_ptr = &static_interpreter;
	input = static_interpreter.input(0);
	output = static_interpreter.output(0);

	xprintf("initial done\n");

	return ercode;
}

void delay_cycles(uint32_t cycles) {
    while (cycles--) {
        __NOP();
    }
}

int cv_run() {
    generate_random_image();

    enable_dwt(); 
    RESET_DWT(); 

    uint32_t start = GET_DWT();
    img_rescale_rgb(random_image, input->data.int8);
    uint32_t end = GET_DWT();
    uint32_t cycles = end - start;
    float time_us = (float)cycles / CPU_FREQ_MHZ; 
    char time_str[CHAR_BUFF_SIZE]; 
    char *time_ptr = _float_to_char(time_us, time_str);  
    xprintf("Memory I/O time: %s us\n", time_ptr);  

    delay_cycles(500000000); 

    start = GET_DWT();
    TfLiteStatus invoke_status = int_ptr->Invoke();
    if(invoke_status != kTfLiteOk) {
        xprintf("Inference failed\n");
        return -1;
    }
	else
		xprintf("invoke pass\n");
    end = GET_DWT();
    cycles = end - start;
    time_us = (float)cycles / CPU_FREQ_MHZ; 
    time_str[CHAR_BUFF_SIZE]; 
    time_ptr = _float_to_char(time_us, time_str);  
    xprintf("Inference time: %s us\n", time_ptr);  

    delay_cycles(500000000); 
    
    start = GET_DWT();
    for (int i = 0; i < OUTPUT_ROWS * OUTPUT_COLS; i++) {
        processed_output[i] = output->data.f[i];
    }
    end = GET_DWT();
    cycles = end - start;
    time_us = (float)cycles / CPU_FREQ_MHZ; 
    time_ptr = _float_to_char(time_us, time_str);  
    xprintf("Memory I/O time: %s us\n", time_ptr); 

    return 0;
}

int cv_deinit()
{
	return 0;
}