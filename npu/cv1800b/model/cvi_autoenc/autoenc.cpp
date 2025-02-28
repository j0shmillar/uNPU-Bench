#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <unistd.h>
#include "core/cvi_tdl_types_mem_internal.h"
#include "cvi_tdl.h"

#include <wiringx.h>

double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}

int main(int argc, char* argv[]) {

    int DUO_LED = 25;

    if(wiringXSetup("milkv_duo", NULL) == -1) {
        wiringXGC();
        return 1;
    }

    if(wiringXValidGPIO(DUO_LED) != 0) {
        printf("Invalid GPIO %d\n", DUO_LED);
    }

    pinMode(DUO_LED, PINMODE_OUTPUT);
    
    digitalWrite(DUO_LED, HIGH);
    sleep(1);
    digitalWrite(DUO_LED, LOW);

    sleep(5);

    digitalWrite(DUO_LED, HIGH);
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
    double end_time = get_time_ms();
    printf("Init time: %.3f ms\n", end_time - start_time);
    digitalWrite(DUO_LED, LOW);

    srand(time(NULL));
    std::vector<uint8_t> random_image(256 * 3);
    for (auto &pixel : random_image) 
    {
        pixel = rand() % 256;
    }

    sleep(5);

    digitalWrite(DUO_LED, HIGH);
    start_time = get_time_ms();
    VIDEO_FRAME_INFO_S fdFrame = {};
    fdFrame.stVFrame.pu8VirAddr[0] = random_image.data();
    fdFrame.stVFrame.u32Width = 256;
    fdFrame.stVFrame.enPixelFormat = PIXEL_FORMAT_RGB_888;
    end_time = get_time_ms();
    printf("Memory I/O time: %.3f ms\n", end_time - start_time);
    digitalWrite(DUO_LED, LOW);

    cvtdl_class_meta_t class_meta = {};

    sleep(5);

    digitalWrite(DUO_LED, HIGH);
    start_time = get_time_ms();
    CVI_TDL_Image_Classification(tdl_handle, &fdFrame, &class_meta);
    end_time = get_time_ms();
    printf("Inference time: %.3f ms\n", end_time - start_time);
    digitalWrite(DUO_LED, LOW);

    CVI_TDL_Free(&class_meta);
    CVI_TDL_DestroyHandle(tdl_handle);
    return ret;
}

