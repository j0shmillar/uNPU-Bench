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
int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <model_path>\n", argv[0]);
        return -1;
    }

    int img_width = 32;
    int img_height = 32;
    int num_classes = 10;
    
    double start_time = get_time_ms();
    cvitdl_handle_t tdl_handle = NULL;
    CVI_S32 ret = CVI_TDL_CreateHandle(&tdl_handle);
    if (ret != CVI_SUCCESS) {
        printf("Create tdl handle failed with %#x!\n", ret);
        return ret;
    }

    std::string model_path = argv[1];
    ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_IMAGE_CLASSIFICATION, model_path.c_str());
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

    printf("Predicted class: %d with confidence: %.3f\n", class_meta.cls[0], class_meta.score[0]);

    CVI_TDL_Free(&class_meta);
    CVI_TDL_DestroyHandle(tdl_handle);
    return ret;
}

