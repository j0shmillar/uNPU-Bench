#include <cstdio>
#include <cstdlib>
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include "WE2_device.h"
#include "board.h"
#include "cvapp.h"
#include "WE2_core.h"
#include "ethosu_driver.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "xprintf.h"
#include "math.h"
#include "yolov1_full_integer_quant_vela.h"
#include "common_config.h"

#define U55_BASE	BASE_ADDR_APB_U55_CTRL

#define INPUT_SIZE_X 96
#define INPUT_SIZE_Y 96
#define INPUT_CHANNELS 3

#define WND_X0 0
#define WND_Y0 0

#define CONV_OUT_H 12
#define CONV_OUT_W 12
#define CONV_OUT_C 12

#define OUT_H 12
#define OUT_W 12
#define SIGMOID_CH 1 
#define SOFTMAX_CH 1 
#define TOTAL_CH (SIGMOID_CH + SOFTMAX_CH)
#define MAX_DETECTIONS (OUT_H * OUT_W)

#define TENSOR_ARENA_BUFSIZE  (400*1024)
__attribute__(( section(".bss.NoInit"))) uint8_t tensor_arena_buf[TENSOR_ARENA_BUFSIZE] __ALIGNED(32);

using namespace std;

namespace {
constexpr int tensor_arena_size = TENSOR_ARENA_BUFSIZE;
uint8_t* tensor_arena = tensor_arena_buf;

struct ethosu_driver ethosu_drv;
tflite::MicroInterpreter *int_ptr = nullptr;
TfLiteTensor* input, *output;
};

static void _arm_npu_irq_handler(void) {
    xprintf("NPU IRQ triggered!\n");  // Debug print
    ethosu_irq_handler(&ethosu_drv);
    xprintf("NPU IRQ handled!\n");  // Debug print after handling
}

static void _arm_npu_irq_init(void) {
    const IRQn_Type ethosu_irqnum = (IRQn_Type)U55_IRQn;
    EPII_NVIC_SetVector(ethosu_irqnum, (uint32_t)_arm_npu_irq_handler);
    NVIC_EnableIRQ(ethosu_irqnum);
}

static int _arm_npu_init(bool security_enable, bool privilege_enable)
{
    int err = 0;

    /* Initialise the IRQ */
    _arm_npu_irq_init();

    /* Initialise Ethos-U55 device */
    const void * ethosu_base_address = (void *)(U55_BASE);

    if (0 != (err = ethosu_init(
                            &ethosu_drv,             /* Ethos-U driver device pointer */
                            ethosu_base_address,     /* Ethos-U NPU's base address. */
                            NULL,       /* Pointer to fast mem area - NULL for U55. */
                            0, /* Fast mem region size. */
							security_enable,                       /* Security enable. */
							privilege_enable))) {                   /* Privilege enable. */
    	xprintf("failed to initalise Ethos-U device\n");
            return err;
        }

    xprintf("Ethos-U55 device initialised\n");

    return 0;
}

int cv_init(bool security_enable, bool privilege_enable)
{
	int ercode = 0;

	if(_arm_npu_init(security_enable, privilege_enable)!=0)
		return -1;

#if (FLASH_XIP_MODEL == 1)
	static const tflite::Model*model = tflite::GetModel((const void *)0x3A180000);
#else
	static const tflite::Model*model = tflite::GetModel((const void *)yolov1_full_integer_quant_vela_tflite);
#endif

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

	static tflite::MicroErrorReporter micro_error_reporter;
	static tflite::MicroMutableOpResolver<1> op_resolver;

	if (kTfLiteOk != op_resolver.AddEthosU()){
		xprintf("Failed to add Arm NPU support to op resolver.");
		return false;
	}

	static tflite::MicroInterpreter static_interpreter(model, op_resolver, (uint8_t*)tensor_arena, tensor_arena_size, &micro_error_reporter);

	if(static_interpreter.AllocateTensors()!= kTfLiteOk) {
		return false;
	}
	int_ptr = &static_interpreter;
	input = static_interpreter.input(0);
	output = static_interpreter.output(0);

	xprintf("initial done\n");

	return ercode;
}

typedef struct {
    float x, y;           // Center coordinates
    float confidence;     // Detection confidence
    float class_score;    // Class probability
    int class_id;        // Predicted class
} detection_t;

void softmax(const float* input, float* output, int length) {
    float max_val = input[0];
    for (int i = 1; i < length; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    float inv_sum = 1.0f / sum;
    for (int i = 0; i < length; i++) {
        output[i] *= inv_sum;
    }
}

float sigmoid(float x) {
    if (x < -5.0f) return 0.0f;
    if (x > 5.0f) return 1.0f;
    return 1.0f / (1.0f + expf(-x));
}

void process_model_output(const float* conv_output, float* processed_output) {
    for (int h = 0; h < OUT_H; h++) {
        for (int w = 0; w < OUT_W; w++) {
            const int out_base_idx = (h * OUT_W + w) * TOTAL_CH;
            const int conv_base_idx = (h * CONV_OUT_W + w) * CONV_OUT_C;

            processed_output[out_base_idx] = sigmoid(conv_output[conv_base_idx]);

            float softmax_input[SOFTMAX_CH];
            softmax_input[0] = conv_output[conv_base_idx + 1];  
            softmax(&softmax_input[0], &processed_output[out_base_idx + SIGMOID_CH], SOFTMAX_CH);
        }
    }
}

void decode_predictions(const float* processed_output, detection_t* detections, int* num_detections, float conf_threshold) {
    *num_detections = 0;

    for (int h = 0; h < OUT_H; h++) {
        for (int w = 0; w < OUT_W; w++) {
            const int base_idx = (h * OUT_W + w) * TOTAL_CH;
            float confidence = processed_output[base_idx];

            if (confidence > conf_threshold) {
                detection_t* det = &detections[*num_detections];

                // Convert grid cell coordinates to image coordinates
                det->x = ((float)w + 0.5f) / OUT_W;
                det->y = ((float)h + 0.5f) / OUT_H;
                det->confidence = confidence;

                // Get class score from softmax output
                det->class_score = processed_output[base_idx + SIGMOID_CH];

                // Assign class ID (for single class, always 0)
                det->class_id = 0;

                (*num_detections)++;
            }
        }
    }
}

void generate_random_input(int8_t* input_data) {
    for (int i = 0; i < INPUT_SIZE_X * INPUT_SIZE_Y * INPUT_CHANNELS; i++) {
        input_data[i] = (rand() % 256) - 128;
    }
}

int cv_run() {

    xprintf("here1\n");

    generate_random_input(input->data.int8);

    if (input == nullptr || input->data.int8 == nullptr) {
        xprintf("Error: input tensor is null\n");
        return -1;
    }

    xprintf("Sample input values: %d, %d, %d...\n", 
            input->data.int8[0], 
            input->data.int8[1], 
            input->data.int8[2]);

    if (!input || !output) {
        xprintf("Error: Failed to allocate input/output tensors\n");
        return -1;
    }

    xprintf("Input type: %d\n", input->type);
    xprintf("Output type: %d\n", output->type);
    xprintf("here2\n");

    // time for invoke
    if (int_ptr->Invoke() != kTfLiteOk) {
        xprintf("invoke fail\n");
        return -1;
    }

    xprintf("here3\n");

    float processed_output[OUT_H * OUT_W * TOTAL_CH];
    detection_t detections[MAX_DETECTIONS];
    int num_detections;
    const float CONFIDENCE_THRESHOLD = 0.5f;

    // time for post-process
    process_model_output((float*)output, processed_output);
    decode_predictions(processed_output, detections, &num_detections, CONFIDENCE_THRESHOLD);

    xprintf("\nDetections found: %d\n", num_detections);
    for (int i = 0; i < num_detections; i++) {
        xprintf("Detection %d: (x=%.3f, y=%.3f) conf=%.3f class=%d score=%.3f\n",
            i,
            detections[i].x,
            detections[i].y,
            detections[i].confidence,
            detections[i].class_id,
            detections[i].class_score);
    }
    xprintf("\n");
    return 0;
}

int cv_deinit() {
    return 0;
}