#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <math.h>
#include "core/cvi_tdl_types_mem_internal.h"
#include "cvi_tdl.h"

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

typedef int32_t q31_t;
typedef int16_t q15_t;

#define Q15_MAX_VALUE   32767
#define Q15_MIN_VALUE   -32768

double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
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

void sigmoid_q15(const q31_t * vec_in, const uint16_t dim_vec, q15_t * p_out)
{
    for (int i = 0; i < dim_vec; i++)
    {
        q31_t val = vec_in[i];
        q15_t sigmoid_val = (val < 0) ? 0 : (val > Q15_MAX_VALUE) ? Q15_MAX_VALUE : val;
        p_out[i] = sigmoid_val;
    }
}

q31_t**** generateArray(int n) {
    q31_t**** array = (q31_t****)malloc(sizeof(q31_t***) * 1);
    array[0] = (q31_t***)malloc(sizeof(q31_t**) * 12);
    
    for (int i = 0; i < 12; i++) {
        array[0][i] = (q31_t**)malloc(sizeof(q31_t*) * 12);
        for (int j = 0; j < 12; j++) {
            array[0][i][j] = (q31_t*)malloc(sizeof(q31_t) * n);
        }
    }

    q31_t value = 0.0f;
    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < 12; j++) {
            for (int k = 0; k < 12; k++) {
                for (int l = 0; l < n; l++) {
                    array[i][j][k][l] = value++;
                }
            }
        }
    }

    return array;
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

    cvtdl_class_meta_t class_meta = {};

    printf("Infer\n");
    start_time = get_time_ms();
    CVI_TDL_Image_Classification(tdl_handle, &fdFrame, &class_meta);
    end_time = get_time_ms();
    printf("Inference time: %.3f ms\n", end_time - start_time);

    q31_t**** array1 = generateArray(2);
    q31_t**** array2 = generateArray(10);
    
    q31_t* class_output = &array1[0][0][0][0]; 
    q31_t* obj_confidence = &array2[0][0][0][0];  

    q15_t softmax_output[OUT_H * OUT_W * NUM_CLASSES];
    q15_t sigmoid_output[OUT_H * OUT_W * NUM_CONFIDENCE];

    start_time = get_time_ms();
    for (int i = 0; i < OUT_H * OUT_W; i++) {
        softmax_q17p14_q15(&class_output[i * NUM_CLASSES], NUM_CLASSES, &softmax_output[i * NUM_CLASSES]);
    }

    for (int i = 0; i < OUT_H * OUT_W; i++) {
        sigmoid_q15(&obj_confidence[i * NUM_CLASSES], NUM_CLASSES, &sigmoid_output[i * NUM_CLASSES]);
    }

    end_time = get_time_ms();
    printf("Post process time: %.3f ms\n", end_time - start_time);

    CVI_TDL_Free(&class_meta);
    CVI_TDL_DestroyHandle(tdl_handle);
    return ret;
}

