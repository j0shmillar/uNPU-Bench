#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"

#include <unistd.h>   
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/fb.h>
#include <time.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("%s <model_path>\n", argv[0]);
        return -1;
    }
    system("RkLunch-stop.sh");
    const char *model_path = argv[1];

    struct timespec start_time, end_time;
    char text[8];
    float fps = 0;

    int model_width  = 256;
    int model_height = 3;
    int channels = 1; // Since the shape is 3x256, assuming a single-channel interpretation

    int ret;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    clock_gettime(CLOCK_MONOTONIC, &start_time);
    init_model(model_path, &rknn_app_ctx);
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double init_time_us = (end_time.tv_sec - start_time.tv_sec) * 1e6 + (end_time.tv_nsec - start_time.tv_nsec) / 1e3;
    printf("Init Time: %.2f microseconds\n", init_time_us);

    int disp_flag = 0;
    int pixel_size = 0;
    size_t screensize = 0;
    int disp_width = 0;
    int disp_height = 0;
    void* framebuffer = NULL; 
    struct fb_fix_screeninfo fb_fix;
    struct fb_var_screeninfo fb_var;

    int framebuffer_fd = 0; // for DMA
    cv::Mat disp;

    disp_height = model_height;
    disp_width = model_width;

    clock_gettime(CLOCK_MONOTONIC, &start_time);
    cv::Mat bgr(disp_height, disp_width, CV_8UC3, cv::Scalar(0, 0, 0));  
    cv::Mat bgr_model_input(model_height, model_width, CV_8UC1); // Using single-channel matrix
    cv::randu(bgr_model_input, cv::Scalar::all(0), cv::Scalar::all(255));
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double memory_time_us = (end_time.tv_sec - start_time.tv_sec) * 1e6 + (end_time.tv_nsec - start_time.tv_nsec) / 1e3;
    printf("Memory I/O Time: %.2f microseconds\n", memory_time_us);

    inference_model(&rknn_app_ctx);

    ret = release_model(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release_model fail! ret=%d\n", ret);
    }

    return 0;
}
