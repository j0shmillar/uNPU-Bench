#include <stdio.h>
#include <unistd.h>
#include "fsl_debug_console.h"
#include "model.h"
#include "timer.h"
#include "math.h"

#define MODEL_IN_W 
#define MODEL_IN_H  
#define MODEL_IN_C
#define OUT_SIZE 

#define Q15_MAX_VALUE   32767
#define Q15_MIN_VALUE   -32768

typedef int32_t q31_t;
typedef int16_t q15_t;

__attribute__((section (".model_input_buffer"))) static uint8_t model_input_buf[MODEL_IN_W*MODEL_IN_H*MODEL_IN_C] = {0};

static float processed_output[OUT_SIZE];

extern "C" {

	void run()
	{
		tensor_dims_t inputDims;
		tensor_type_t inputType;
		uint8_t* inputData;

		tensor_dims_t outputDims;
		tensor_type_t outputType;
		uint8_t* outputData;
		size_t arenaSize;

		if (MODEL_Init() != kStatus_Success)
		{
			PRINTF("Failed initializing model");
			for (;;) {}
		}

		size_t usedSize = MODEL_GetArenaUsedBytes(&arenaSize);
		PRINTF("\r\n%d/%d kB (%0.2f%%) tensor arena used\r\n", usedSize / 1024, arenaSize / 1024, 100.0*usedSize/arenaSize);

		inputData = MODEL_GetInputTensorData(&inputDims, &inputType);
		outputData = MODEL_GetOutputTensorData(&outputDims, &outputType);

		TfLiteTensor* outputTensor = MODEL_GetOutputTensor(0);
		
		while(1)
		{
			uint8_t *pOut = inputData;
			for (int i = 0; i < MODEL_IN_W; i++) {
				for (int j = 0; j < MODEL_IN_H; j++) {
					for (int c = 0; c < MODEL_IN_C; c++) {
						*pOut++ = 0;
					}
				}
			}
			uint8_t *buf = 0;
			memset(inputData,0,inputDims.data[1]*inputDims.data[2]*inputDims.data[3]);
			buf = inputData + (inputData,inputDims.data[1] - MODEL_IN_H) /2 * MODEL_IN_W * MODEL_IN_C;
			memcpy(buf, model_input_buf, MODEL_IN_W*MODEL_IN_H*MODEL_IN_C);

			MODEL_RunInference();

			size_t output_size = OUT_SIZE * sizeof(int8_t);

			memcpy(processed_output, outputData, output_size);
		}

	}
}


