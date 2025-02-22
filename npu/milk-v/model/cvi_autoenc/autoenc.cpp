#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <math.h>
#include "core/cvi_tdl_types_mem_internal.h"
#include "cvi_tdl.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <model_path>\n", argv[0]);
        return -1;
    }

    int img_width = 80;
    int img_height = 80;
    int num_classes = 3;
    
    cvitdl_handle_t tdl_handle = NULL;
    CVI_S32 ret = CVI_TDL_CreateHandle(&tdl_handle);
    if (ret != CVI_SUCCESS) {
        printf("Create tdl handle failed with %#x!\n", ret);
        return ret;
    }

    std::string model_path = argv[1];
    ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_RESNET, model_path.c_str());
    if (ret != CVI_SUCCESS) {
        printf("Open model failed %#x!\n", ret);
        return ret;
    }

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

    cvtdl_class_t class_meta = {};
    CVI_TDL_AutoEnc(tdl_handle, &fdFrame, &class_meta);

    printf("Predicted class: %d with confidence: %.3f\n", class_meta.best_class, class_meta.confidence);

    CVI_TDL_Free(&class_meta);
    CVI_TDL_DestroyHandle(tdl_handle);
    return ret;
}
