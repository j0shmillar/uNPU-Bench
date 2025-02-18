#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <math.h>
#include "core/cvi_tdl_types_mem_internal.h"
#include "cvi_tdl.h"

// TODO add timing + post process

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
	    printf("here3\n");
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

int main(int argc, char* argv[]) {
    int img_width = 96;
    int img_height = 96;

    cvitdl_handle_t tdl_handle = NULL;
    CVI_S32 ret = CVI_TDL_CreateHandle(&tdl_handle);
    if (ret != CVI_SUCCESS) {
        printf("Create tdl handle failed with %#x!\n", ret);
        return ret;
    }

    std::string model_path = argv[1];
    ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_YOLO, model_path.c_str());
    if (ret != CVI_SUCCESS) {
        printf("Open model failed %#x!\n", ret);
        return ret;
    }
    
    CVI_TDL_SetModelThreshold(tdl_handle, CVI_TDL_SUPPORTED_MODEL_YOLO, 0.5);
    CVI_TDL_SetModelNmsThreshold(tdl_handle, CVI_TDL_SUPPORTED_MODEL_YOLO, 0.5);
    
    srand(time(NULL));
    std::vector<uint8_t> random_image(img_width * img_height * 3);
    for (auto &pixel : random_image) {
        pixel = rand() % 256;
    }

    VIDEO_FRAME_INFO_S fdFrame = {};
    fdFrame.stVFrame.pu8VirAddr[0] = random_image.data();
    fdFrame.stVFrame.u32Width = img_width;
    fdFrame.stVFrame.u32Height = img_height;
    fdFrame.stVFrame.enPixelFormat = PIXEL_FORMAT_RGB_888;

    cvtdl_object_t obj_meta = {0};
    CVI_TDL_Yolo(tdl_handle, &fdFrame, &obj_meta);

    detection_t detections[MAX_DETECTIONS];
    int num_detections = 0;

    for (CVI_U32 i = 0; i < obj_meta.size; i++) {
        if (num_detections >= MAX_DETECTIONS) break;
        
        cvtdl_bbox_t *bbox = &obj_meta.info[i].bbox;
        detection_t *det = &detections[num_detections++];
        
        det->x = (bbox->x1 + bbox->x2) / 2.0f;
        det->y = (bbox->y1 + bbox->y2) / 2.0f;
        det->confidence = 1.0f;  // Placeholder value, since `conf` is not defined
        det->class_id = obj_meta.info[i].classes; // Using `classes` instead of `class_id`
        
        for (int j = 0; j < NUM_CLASSES; j++) {
            det->class_probs[j] = 0.0f;  // No equivalent `class_probs` in `cvtdl_object_info_t`
        }
    }
    
    printf("\nDetections found: %d\n", num_detections);
    for (int i = 0; i < num_detections; i++) {
        printf("Detection %d: (x=%.3f, y=%.3f) conf=%.3f class=%d\n",
               i,
               detections[i].x,
               detections[i].y,
               detections[i].confidence,
               detections[i].class_id);
    }

    CVI_TDL_Free(&obj_meta);
    CVI_TDL_DestroyHandle(tdl_handle);
    return ret;
}

