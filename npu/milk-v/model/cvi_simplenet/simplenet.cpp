#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <math.h>
#include "core/cvi_tdl_types_mem_internal.h"
#include "cvi_tdl.h"

typedef int32_t q31_t;
typedef int16_t q15_t;

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

double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}

q31_t** generateArray() {
    q31_t** array = (q31_t**)malloc(sizeof(q31_t*) * 1);
    array[0] = (q31_t*)malloc(sizeof(q31_t) * 100);  

    q31_t value = 0;
    for (int i = 0; i < 100; i++) {
        array[0][i] = value++; 
    }

    return array;
}


int main(int argc, char* argv[]) {
    int img_width = 32;
    int img_height = 32;
    int num_classes = 100;
    
    double start_time = get_time_ms();
    cvitdl_handle_t tdl_handle = NULL;
    CVI_S32 ret = CVI_TDL_CreateHandle(&tdl_handle);
    if (ret != CVI_SUCCESS) {
        printf("Create tdl handle failed with %#x!\n", ret);
        return ret;
    }

    std::string model_path = argv[1];
    std::cout << "Model path: " << model_path << std::endl;
    ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_YOLO, model_path.c_str());
    if (ret != CVI_SUCCESS) {
        printf("Open model failed %#x!\n", ret);
        return ret;
    }
    double end_time = get_time_ms();
    printf("Init time: %.3f ms\n", end_time - start_time);

    start_time = get_time_ms();
    srand(time(NULL));
    std::vector<uint8_t> random_image(img_width * img_height * 3);
    for (auto &pixel : random_image) 
    {
        pixel = rand() % 256;
    }
    VIDEO_FRAME_INFO_S fdFrame = {};
    fdFrame.stVFrame.pu8VirAddr[0] = random_image.data();
    fdFrame.stVFrame.u32Width = img_width;
    fdFrame.stVFrame.u32Height = img_height;
    fdFrame.stVFrame.enPixelFormat = PIXEL_FORMAT_RGB_888;
    end_time = get_time_ms();
    printf("Memory I/O time: %.3f ms\n", end_time - start_time);

    cvtdl_class_meta_t class_meta = {};

    start_time = get_time_ms();
    CVI_TDL_Image_Classification(tdl_handle, &fdFrame, &class_meta);
    end_time = get_time_ms();
    printf("Inference time: %.3f ms\n", end_time - start_time);

    start_time = get_time_ms();
    q31_t** array = generateArray();
    q15_t p_out[100];
    softmax_q17p14_q15(array[0], 100, p_out);
    end_time = get_time_ms();
    printf("Post-processing time: %.3f ms\n", end_time - start_time);

    CVI_TDL_Free(&class_meta);
    CVI_TDL_DestroyHandle(tdl_handle);
    return ret;
}
