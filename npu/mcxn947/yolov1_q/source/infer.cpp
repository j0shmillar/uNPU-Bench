#include <stdio.h>
#include "fsl_debug_console.h"
#include "image.h"
#include "image_utils.h"
#include "model.h"
#include "output_postproc.h"
#include "timer.h"
#include "math.h"

extern "C" {

#define MODEL_IN_W	96
#define MODEL_IN_H  96
#define MODEL_IN_C	3
#define MODEL_IN_COLOR_BGR 0


__attribute__((section (".model_input_buffer"))) static uint8_t model_input_buf[MODEL_IN_W*MODEL_IN_H*MODEL_IN_C] = {0};

uint32_t s_Us = 0;
volatile uint8_t g_isImgBufReady = 0;
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

void Rgb565StridedToBgr888(const uint16_t* pIn, int srcW, int wndW, int wndH, int wndX0, int wndY0,
	uint8_t* p888, int stride, uint8_t isSub128) {
	const uint16_t* pSrc = pIn;
	uint32_t datIn, datOut, datOuts[3];
	uint8_t* p888out = p888;
	for (int y = wndY0,y1=(wndH-wndY0)/stride-wndY0; y < wndH; y += stride,y1--) {
		pSrc = pIn + srcW * y + wndX0;
		//p888out = p888 + y1*wndW*3/stride;
		for (int x = 0; x < wndW; x += stride * 4) {
			datIn = pSrc[0];
			pSrc += stride;
			datOuts[0] = (datIn & 31) << 3 | (datIn & 63 << 5) << 5 | (datIn & 31 << 11) << 8;
			// datOuts[0] = (datIn & 31) << 19| (datIn & 63 << 5) << 5 | ((datIn>>8) & 0xf8);

			datIn = pSrc[0];
			pSrc += stride;
			datOut = (datIn & 31) << 3 | (datIn & 63 << 5) << 5 | (datIn & 31 << 11) << 8;
			// datOut = (datIn & 31) << 19| (datIn & 63 << 5) << 5 | ((datIn>>8) & 0xf8);
			datOuts[0] |= datOut << 24;
			datOuts[1] = datOut >> 8;

			datIn = pSrc[0];
			pSrc += stride;
			datOut = (datIn & 31) << 3 | (datIn & 63 << 5) << 5 | (datIn & 31 << 11) << 8;
			// datOut = (datIn & 31) << 19| (datIn & 63 << 5) << 5 | ((datIn>>8) & 0xf8);
			datOuts[1] |= (datOut << 16) & 0xFFFF0000;
			datOuts[2] = datOut >> 16;

			datIn = pSrc[0];
			pSrc += stride;
			datOut = (datIn & 31) << 3 | (datIn & 63 << 5) << 5 | (datIn & 31 << 11) << 8;
			// datOut = (datIn & 31) << 19| (datIn & 63 << 5) << 5 | ((datIn>>8) & 0xf8);

			datOuts[2] |= datOut << 8;

			if (isSub128) {
				// subtract 128 bytewisely, equal to XOR with 0x80
				datOuts[0] ^= 0x80808080;
				datOuts[1] ^= 0x80808080;
				datOuts[2] ^= 0x80808080;
			}
			memcpy(p888out, datOuts, 3 * 4);
			p888out += 3 * 4;
		}
	}
}

void Rgb565StridedToRgb888(const uint16_t* pIn, int srcW, int wndW, int wndH, int wndX0, int wndY0,
	uint8_t* p888, int stride, uint8_t isSub128) {
	const uint16_t* pSrc = pIn;
	uint32_t datIn, datOut, datOuts[3];
	uint8_t* p888out = p888;

	for (int y = wndY0,y1=(wndH-wndY0)/stride-wndY0; y < wndH; y += stride,y1--) {
		pSrc = pIn + srcW * y + wndX0;

		//p888out = p888 + y1*wndW*3/stride;
		for (int x = 0; x < wndW; x += stride * 4) {
			datIn = pSrc[0];
			pSrc += stride;
			// datOuts[0] = (datIn & 31) << 3 | (datIn & 63 << 5) << 5 | (datIn & 31 << 11) << 8;
			datOuts[0] = (datIn & 31) << 19| (datIn & 63 << 5) << 5 | ((datIn>>8) & 0xf8);

			datIn = pSrc[0];
			pSrc += stride;
			// datOut = (datIn & 31) << 3 | (datIn & 63 << 5) << 5 | (datIn & 31 << 11) << 8;
			datOut = (datIn & 31) << 19| (datIn & 63 << 5) << 5 | ((datIn>>8) & 0xf8);
			datOuts[0] |= datOut << 24;
			datOuts[1] = datOut >> 8;

			datIn = pSrc[0];
			pSrc += stride;
			// datOut = (datIn & 31) << 3 | (datIn & 63 << 5) << 5 | (datIn & 31 << 11) << 8;
			datOut = (datIn & 31) << 19| (datIn & 63 << 5) << 5 | ((datIn>>8) & 0xf8);
			datOuts[1] |= (datOut << 16) & 0xFFFF0000;
			datOuts[2] = datOut >> 16;

			datIn = pSrc[0];
			pSrc += stride;
			// datOut = (datIn & 31) << 3 | (datIn & 63 << 5) << 5 | (datIn & 31 << 11) << 8;
			datOut = (datIn & 31) << 19| (datIn & 63 << 5) << 5 | ((datIn>>8) & 0xf8);

			datOuts[2] |= datOut << 8;

			if (isSub128) {
				// subtract 128 bytewisely, equal to XOR with 0x80
				datOuts[0] ^= 0x80808080;
				datOuts[1] ^= 0x80808080;
				datOuts[2] ^= 0x80808080;
			}
			memcpy(p888out, datOuts, 3 * 4);
			p888out += 3 * 4;
		}
	}
}

void ezh_copy_slice_to_model_input(uint32_t idx, uint32_t cam_slice_buffer, uint32_t cam_slice_width, uint32_t cam_slice_height, uint32_t max_idx)
{
	static uint8_t* pCurDat;
	uint32_t curY;
	uint32_t s_imgStride = cam_slice_width / MODEL_IN_W;


	if (idx > max_idx)
		return;
	//uint32_t ndx = max_idx -1 - idx;
	uint32_t ndx = idx;
	curY = ndx * cam_slice_height;
	int wndY = (s_imgStride - (curY - WND_Y0) % s_imgStride) % s_imgStride;


	if (idx +1 >= max_idx)
		g_isImgBufReady = 1;

	pCurDat = model_input_buf + 3 * ((cam_slice_height * ndx + wndY) * cam_slice_width / s_imgStride / s_imgStride);

	if (curY + cam_slice_height >= WND_Y0){

		if (MODEL_IN_COLOR_BGR == 1) {
			Rgb565StridedToBgr888((uint16_t*)cam_slice_buffer, cam_slice_width, cam_slice_width, cam_slice_height, WND_X0, wndY, pCurDat, s_imgStride, 0);
		}else {
			Rgb565StridedToRgb888((uint16_t*)cam_slice_buffer, cam_slice_width, cam_slice_width, cam_slice_height, WND_X0, wndY, pCurDat, s_imgStride, 0);
		}
	}
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

void decode_predictions(const float* processed_output, 
                       detection_t* detections,
                       int* num_detections,
                       float conf_threshold) {
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
                det->class_id = 0;  // For single class, always 0
                
                (*num_detections)++;
            }
        }
    }
}

void infer() {
    tensor_dims_t inputDims;
    tensor_type_t inputType;
    int8_t* inputData;

    tensor_dims_t outputDims;
    tensor_type_t outputType;
    int8_t* outputData;
    size_t arenaSize;

    PRINTF("Initializing model...\n");
    if (MODEL_Init() != kStatus_Success) {
        PRINTF("Failed initializing model\n");
        for (;;) {}
    }

    size_t usedSize = MODEL_GetArenaUsedBytes(&arenaSize);
    PRINTF("\r\n%d/%d kB (%0.2f%%) tensor arena used\r\n", usedSize / 1024, arenaSize / 1024, 100.0*usedSize/arenaSize);

    inputData = MODEL_GetInputTensorData(&inputDims, &inputType);
    outputData = MODEL_GetOutputTensorData(&outputDims, &outputType);

    // Allocate buffers for intermediate and final results
    float processed_output[OUT_H * OUT_W * TOTAL_CH];
    detection_t detections[MAX_DETECTIONS];
    int num_detections;
    const float CONFIDENCE_THRESHOLD = 0.5f;

    while(1) {
        auto startTime = TIMER_GetTimeInUS();
        int8_t *pOut = inputData;
        for (int i = 0; i < MODEL_IN_W; i++) {
            for (int j = 0; j < MODEL_IN_H; j++) {
                for (int c = 0; c < MODEL_IN_C; c++) {
                    *pOut++ = 0;
                }
            }
        }
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
        
        process_model_output((float*)outputData, processed_output);
        
        decode_predictions(processed_output, detections, &num_detections, CONFIDENCE_THRESHOLD);
        
        endTime = TIMER_GetTimeInUS();
        dt = endTime - startTime;
        s_Us = (uint32_t)dt;
        PRINTF("Post-processing: %d us\n", s_Us);

        PRINTF("\nDetections found: %d\n", num_detections);
        for (int i = 0; i < num_detections; i++) {
            PRINTF("Detection %d: (x=%.3f, y=%.3f) conf=%.3f class=%d score=%.3f\n",
                   i,
                   detections[i].x,
                   detections[i].y,
                   detections[i].confidence,
                   detections[i].class_id,
                   detections[i].class_score);
        }
        PRINTF("\n");
        
        uint64_t wait_start = TIMER_GetTimeInUS();
        while (TIMER_GetTimeInUS() - wait_start < 30000000) {}
    }
}

}


