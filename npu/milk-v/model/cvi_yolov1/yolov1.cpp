#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <math.h>
#include "core/cvi_tdl_types_mem_internal.h"
#include "cvi_tdl.h"

double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}

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
    const int confidence_stride = grid_size * grid_size * 2;  
    
    for (int y = 0; y < grid_size; y++) {
        for (int x = 0; x < grid_size; x++) {
            int grid_idx = y * grid_size + x;
            int out_idx = grid_idx * OUTPUT_STRIDE;
            
            for (int i = 0; i < NUM_CONFIDENCE; i++) {
                float conf = output_array[grid_idx * 2 + i]; 
                final_output[out_idx + i] = sigmoid(conf);
            }
            
            float class_inputs[NUM_CLASSES];
            for (int i = 0; i < NUM_CLASSES; i++) {
                class_inputs[i] = output_array[confidence_stride + grid_idx * NUM_CLASSES + i];
            }
            
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

    double start_time = get_time_ms();
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
    double end_time = get_time_ms();
    printf("Init time: %.3f ms\n", end_time - start_time);

    std::vector<uint8_t> random_image(img_width * img_height * 3);
    for (auto &pixel : random_image) 
    {
        pixel = rand() % 256;
    }
    
    start_time = get_time_ms();
    srand(time(NULL));
    VIDEO_FRAME_INFO_S fdFrame = {};
    fdFrame.stVFrame.pu8VirAddr[0] = random_image.data();
    fdFrame.stVFrame.u32Width = img_width;
    fdFrame.stVFrame.u32Height = img_height;
    fdFrame.stVFrame.enPixelFormat = PIXEL_FORMAT_RGB_888;
    end_time = get_time_ms();
    printf("Memory I/O time: %.3f ms\n", end_time - start_time);

    cvtdl_object_t obj_meta = {0};

    start_time = get_time_ms();
    CVI_TDL_Yolo(tdl_handle, &fdFrame, &obj_meta);
    end_time = get_time_ms();
    printf("Inference time: %.3f ms\n", end_time - start_time);

    start_time = get_time_ms();
    float processed_output[MAX_DETECTIONS * OUTPUT_STRIDE];
    process_yolo_output(obj_meta.info[0].data, processed_output, GRID_SIZE);

    detection_t detections[MAX_DETECTIONS];
    int num_detections = 0;
    extract_detections(processed_output, detections, &num_detections, 0.5, GRID_SIZE);
    end_time = get_time_ms();
    printf("Post process time: %.3f ms\n", end_time - start_time);

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
