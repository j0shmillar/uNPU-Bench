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
#define NUM_CLASSES 2  // Based on the output tensor shape shown
#define NUM_CONFIDENCE 10 // Based on the slice operations shown
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

	void process_yolo_output(const float* output_array, float* final_output, int grid_size) {
		// Each grid cell has:
		// - 2 confidence scores from first slice
		// - 10 class probabilities from second slice
		const int confidence_stride = grid_size * grid_size * 2;  // Size of confidence tensor
		
		// Process each grid cell
		for (int y = 0; y < grid_size; y++) {
			for (int x = 0; x < grid_size; x++) {
				int grid_idx = y * grid_size + x;
				int out_idx = grid_idx * OUTPUT_STRIDE;
				
				// Get confidence scores (first 2 values per grid cell)
				for (int i = 0; i < NUM_CONFIDENCE; i++) {
					float conf = output_array[grid_idx * 2 + i];  // From first part of output
					final_output[out_idx + i] = sigmoid(conf);
				}
				
				// Get class probabilities (next 10 values per grid cell)
				float class_inputs[NUM_CLASSES];
				for (int i = 0; i < NUM_CLASSES; i++) {
					class_inputs[i] = output_array[confidence_stride + grid_idx * NUM_CLASSES + i];
				}
				
				// Apply softmax to class probabilities
				softmax(class_inputs, &final_output[out_idx + NUM_CONFIDENCE], NUM_CLASSES);
			}
		}
	}


	void extract_detections(const float* processed_output, 
						detection_t* detections,
						int* num_detections,
						float confidence_threshold,
						int grid_size) {
		*num_detections = 0;
		
		for (int y = 0; y < grid_size; y++) {
			for (int x = 0; x < grid_size; x++) {
				int grid_idx = y * grid_size + x;
				int out_idx = grid_idx * OUTPUT_STRIDE;
				
				float conf1 = processed_output[out_idx];
				float conf2 = processed_output[out_idx + 1];
				
				if (conf1 > confidence_threshold) {
					detection_t* det = &detections[*num_detections];
					
					det->x = ((float)x + 0.5f) / grid_size;
					det->y = ((float)y + 0.5f) / grid_size;
					det->confidence = conf1;
					
					float max_prob = 0.0f;
					for (int i = 0; i < NUM_CLASSES; i++) {
						float prob = processed_output[out_idx + NUM_CONFIDENCE + i];
						det->class_probs[i] = prob;  
						if (prob > max_prob) {
							max_prob = prob;
							det->class_id = i;
						}
					}
					
					(*num_detections)++;
				}
				
				if (conf2 > confidence_threshold) {
					detection_t* det = &detections[*num_detections];
					
					det->x = ((float)x + 0.5f) / grid_size;
					det->y = ((float)y + 0.5f) / grid_size;
					det->confidence = conf2;
					
					float max_prob = 0.0f;
					for (int i = 0; i < NUM_CLASSES; i++) {
						float prob = processed_output[out_idx + NUM_CONFIDENCE + i];
						det->class_probs[i] = prob;  
						if (prob > max_prob) {
							max_prob = prob;
							det->class_id = i;
						}
					}
					
					(*num_detections)++;
				}
			}
		}
	}

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


