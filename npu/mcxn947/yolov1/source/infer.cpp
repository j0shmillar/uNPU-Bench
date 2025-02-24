#include <stdio.h>
#include "fsl_debug_console.h"
#include "model.h"
#include "timer.h"
#include "math.h"

#define MODEL_IN_W 96
#define MODEL_IN_H  96
#define MODEL_IN_C 3
#define MODEL_IN_COLOR_BGR 0

#define GRID_SIZE 12
#define NUM_CLASSES 2  
#define NUM_CONFIDENCE 10 
#define OUTPUT_STRIDE (NUM_CLASSES + NUM_CONFIDENCE)

#define CONV_OUT_H 12
#define CONV_OUT_W 12
#define CONV_OUT_C 12

#define OUT_H 12
#define OUT_W 12
#define SIGMOID_CH 1 
#define SOFTMAX_CH 1 
#define TOTAL_CH (SIGMOID_CH + SOFTMAX_CH)
#define MAX_DETECTIONS (OUT_H * OUT_W)

typedef struct {
	float x, y;    
	float confidence;    
	float class_probs[NUM_CLASSES];  
	int class_id;     
} detection_t;

__attribute__((section (".model_input_buffer"))) static uint8_t model_input_buf[MODEL_IN_W*MODEL_IN_H*MODEL_IN_C] = {0};

static float processed_output[GRID_SIZE * GRID_SIZE * OUTPUT_STRIDE];
static detection_t detections[GRID_SIZE * GRID_SIZE * 2]; // TODO increase?

extern "C" {

	uint32_t s_Us = 0;
	volatile int8_t g_isImgBufReady = 0;
	#define WND_X0 0
	#define WND_Y0 0

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
			auto startTime = TIMER_GetTimeInUS();
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

			auto endTime = TIMER_GetTimeInUS();
			auto dt = endTime - startTime;
			s_Us = (uint32_t)dt;
			PRINTF("Memory I/O: %d us\n", s_Us);

			startTime = TIMER_GetTimeInUS();
			MODEL_RunInference();
			endTime = TIMER_GetTimeInUS();
			dt = endTime - startTime;
			s_Us = (uint32_t)dt;
			PRINTF("Inference: %d us\n", s_Us);

			startTime = TIMER_GetTimeInUS();

			int num_detections;
				
			process_yolo_output((float*)outputData, processed_output, GRID_SIZE);

			float confidence_threshold = 0.5f;
			extract_detections(processed_output, detections, &num_detections, confidence_threshold, GRID_SIZE);

			PRINTF("\nDetections found: %d\n", num_detections);
			for (int i = 0; i < num_detections; i++) {
				PRINTF("Detection %d: (x=%.3f, y=%.3f) conf=%.3f class=%d score=%.3f\n",
					i,
					detections[i].x,
					detections[i].y,
					detections[i].confidence,
					detections[i].class_id,
					detections[i].class_probs[detections[i].class_id]);
			}

			endTime = TIMER_GetTimeInUS();
			dt = endTime - startTime;
			s_Us = (uint32_t)dt;
			PRINTF("Post-processing: %d us\n", s_Us);
		}

	}
}


