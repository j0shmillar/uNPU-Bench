#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "esp_timer.h"
#include "esp_random.h"

#include "main_functions.h"
#include "model.h"

#define INPUT_SIZE_X 32
#define INPUT_SIZE_Y 32
#define INPUT_CHANNELS 3
#define OUTPUT_CLASSES 100

static float processed_output[OUTPUT_CLASSES];

#define Q15_MAX_VALUE   32767
#define Q15_MIN_VALUE   -32768

typedef int32_t q31_t;
typedef int16_t q15_t;

namespace 
{
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 2000;
uint8_t tensor_arena[kTensorArenaSize];
}  

void softmax_q17p14_q15(const q31_t * vec_in, const uint16_t dim_vec, q15_t * p_out)
{
    q31_t     sum;
    int16_t   i;
    uint8_t   shift;
    q31_t     base;
    base = -1 * 0x80000000;

    for (i = 0; i < dim_vec; i++)
    {
        if (vec_in[i] > base)
        {
            base = vec_in[i];
        }
    }

    base = base - (16<<14);

    sum = 0;

    for (i = 0; i < dim_vec; i++)
    {
        if (vec_in[i] > base)
        {
            shift = (uint8_t)((8192 + vec_in[i] - base) >> 14);
            sum += (0x1 << shift);
        }
    }


    /* This is effectively (0x1 << 32) / sum */
    int64_t div_base = 0x100000000LL;
    int32_t output_base = (int32_t)(div_base / sum);
    int32_t out;

    for (i = 0; i < dim_vec; i++)
    {
        if (vec_in[i] > base)
        {
            shift = (uint8_t)(17+((8191 + base - vec_in[i]) >> 14));

            out = (output_base >> shift);

            if (out > 32767)
            	out = 32767;

            p_out[i] = (q15_t)out;


        } else
        {
            p_out[i] = 0;
        }
    }

}

void generate_random_image() {
    for (int i = 0; i < 32 * 32 * 3; i++) {
        input->data.int8[i] = (int8_t)(esp_random() % 256 - 128);
    }
}

q31_t** generateArray() {
    q31_t** array = (q31_t**)malloc(sizeof(q31_t*) * 1);
    array[0] = (q31_t*)malloc(sizeof(q31_t) * OUTPUT_CLASSES);  

    q31_t value = 0;
    for (int i = 0; i < OUTPUT_CLASSES; i++) {
        array[0][i] = value++; 
    }

    return array;
}

void setup() 
{
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) 
  {
    MicroPrintf("Model provided is schema version %d not equal to supported version %d.", model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::MicroMutableOpResolver<15> resolver;

  resolver.AddPad();
  resolver.AddConcatenation();
  resolver.AddSlice();
  resolver.AddResizeNearestNeighbor();
  resolver.AddTranspose();
  resolver.AddSplit();
  resolver.AddConv2D();
  resolver.AddRelu();
  resolver.AddMul();
  resolver.AddAdd();
  resolver.AddMaxPool2D();
  resolver.AddReshape();
  resolver.AddSub();
  resolver.AddLogistic();

  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) 
  {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
}

void run() 
{

  vTaskDelay(pdMS_TO_TICKS(5000));
 
  int64_t start_time = esp_timer_get_time();
  generate_random_image();
  int64_t end_time = esp_timer_get_time();
  printf("Memory I/O time: %lld us\n", end_time - start_time);

  vTaskDelay(pdMS_TO_TICKS(5000));

  start_time = esp_timer_get_time();
  TfLiteStatus invoke_status = interpreter->Invoke();
  end_time = esp_timer_get_time();
  printf("Inference time: %lld us\n", end_time - start_time);
  if (invoke_status != kTfLiteOk) 
  {
    MicroPrintf("Invoke failed\n");
    return;
  }
  end_time = esp_timer_get_time();
  printf("Infer time: %lld us\n", end_time - start_time);

  vTaskDelay(pdMS_TO_TICKS(5000));

  start_time = esp_timer_get_time();
  size_t output_size = OUTPUT_CLASSES * sizeof(int8_t);
  memcpy(processed_output, output->data.int8, output_size);
  end_time = esp_timer_get_time();
  printf("Memory I/O time: %lld us\n", end_time - start_time);

  q31_t** array = generateArray();
  q15_t p_out[OUTPUT_CLASSES];

  vTaskDelay(pdMS_TO_TICKS(5000));

  start_time = esp_timer_get_time();
  softmax_q17p14_q15(array[0], OUTPUT_CLASSES, p_out);
  end_time = esp_timer_get_time();
  printf("Post-processing time: %lld us\n", end_time - start_time);

}
