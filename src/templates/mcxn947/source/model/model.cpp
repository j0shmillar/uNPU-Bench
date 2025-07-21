#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "fsl_debug_console.h"
#include "model.h"

static const tflite::Model* s_model = nullptr;
static tflite::MicroInterpreter* s_interpreter = nullptr;

extern tflite::MicroOpResolver &MODEL_GetOpsResolver();
extern uint8_t npu_model_data[];
constexpr int kTensorArenaSize = (330) * 1024;
static uint8_t s_tensorArena[kTensorArenaSize] __ALIGNED(16);

status_t MODEL_Init(void)
{
    s_model = tflite::GetModel(npu_model_data);
    if (s_model->version() != TFLITE_SCHEMA_VERSION)
    {
        PRINTF("Model provided is schema version %d not equal "
               "to supported version %d.",
               s_model->version(), TFLITE_SCHEMA_VERSION);
        return kStatus_Fail;
    }

    tflite::MicroOpResolver &micro_op_resolver = MODEL_GetOpsResolver();

    static tflite::MicroInterpreter static_interpreter(
        s_model, micro_op_resolver, s_tensorArena, kTensorArenaSize);
    s_interpreter = &static_interpreter;

    TfLiteStatus allocate_status = s_interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk)
    {
        PRINTF("AllocateTensors() failed");
        return kStatus_Fail;
    }

    return kStatus_Success;
}

status_t MODEL_RunInference(void)
{
    if (s_interpreter->Invoke() != kTfLiteOk)
    {
        PRINTF("Invoke failed!\r\n");
        return kStatus_Fail;
    }

    return kStatus_Success;
}

uint8_t* GetTensorData(TfLiteTensor* tensor, tensor_dims_t* dims, tensor_type_t* type)
{
    switch (tensor->type)
    {
        case kTfLiteFloat32:
            *type = kTensorType_FLOAT32;
            break;
        case kTfLiteUInt8:
            *type = kTensorType_UINT8;
            break;
        case kTfLiteInt8:
            *type = kTensorType_INT8;
            break;
        default:
            assert("Unknown input tensor data type");
    };

    dims->size = tensor->dims->size;
    assert(dims->size <= MAX_TENSOR_DIMS);
    for (int i = 0; i < tensor->dims->size; i++)
    {
        dims->data[i] = tensor->dims->data[i];
    }

    return tensor->data.uint8;
}

uint8_t* MODEL_GetInputTensorData(tensor_dims_t* dims, tensor_type_t* type)
{
    TfLiteTensor* inputTensor = s_interpreter->input(0);

    return GetTensorData(inputTensor, dims, type);
}

uint8_t* MODEL_GetOutputTensorData(tensor_dims_t* dims, tensor_type_t* type)
{
    TfLiteTensor* outputTensor = s_interpreter->output(0);

    return GetTensorData(outputTensor, dims, type);
}

void MODEL_ConvertInput(uint8_t* data, tensor_dims_t* dims, tensor_type_t type)
{
    int size = dims->data[2] * dims->data[1] * dims->data[3];
    switch (type)
    {
        case kTensorType_UINT8:
            break;
        case kTensorType_INT8:
            for (int i = size - 1; i >= 0; i--)
            {
                reinterpret_cast<int8_t*>(data)[i] =
                    static_cast<int>(data[i]) - 127;
            }
            break;
        case kTensorType_FLOAT32:
            for (int i = size - 1; i >= 0; i--)
            {
                reinterpret_cast<float*>(data)[i] =
                    (static_cast<int>(data[i]) - 0) / 1;
            }
            break;
        default:
            assert("Unknown input tensor data type");
    }
}

size_t MODEL_GetArenaUsedBytes(size_t *pMaxSize) {
    if (pMaxSize) {
        pMaxSize[0] = sizeof(s_tensorArena);
    }
    return s_interpreter->arena_used_bytes();
}

TfLiteTensor* MODEL_GetOutputTensor(uint32_t idx)
{
    return s_interpreter->output(idx);
}
