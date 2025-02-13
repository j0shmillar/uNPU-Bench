#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "yolov1.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"

#include <unistd.h>   
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/fb.h>
#include <time.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "dma_alloc.cpp"

#define USE_DMA 0

void mapCoordinates(cv::Mat input, cv::Mat output, int *x, int *y) {	
	float scaleX = (float)output.cols / (float)input.cols; 
	float scaleY = (float)output.rows / (float)input.rows;
    
    *x = (int)((float)*x / scaleX);
    *y = (int)((float)*y / scaleY);
}


int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("%s <yolov1 model_path>\n", argv[0]);
        return -1;
    }
    system("RkLunch-stop.sh");
    const char *model_path = argv[1];

    struct timespec start_time, end_time;
    char text[8];
    float fps = 0;

    int model_width    = 96;
    int model_height   = 96;
    int channels = 3;

    int ret;
    rknn_app_context_t rknn_app_ctx;
    object_detect_result_list od_results;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    init_yolov1_model(model_path, &rknn_app_ctx);

    int disp_flag = 0;
    int pixel_size = 0;
    size_t screensize = 0;
    int disp_width = 0;
    int disp_height = 0;
    void* framebuffer = NULL; 
    struct fb_fix_screeninfo fb_fix;
    struct fb_var_screeninfo fb_var;

    int framebuffer_fd = 0; //for DMA
    cv::Mat disp;

    disp_height = 96;
    disp_width = 96;

    cv::Mat bgr(disp_height, disp_width, CV_8UC3, cv::Scalar(0, 0, 0));  

    cv::Mat bgr_model_input(model_height, model_width, CV_8UC3);
    cv::randu(bgr_model_input, cv::Scalar::all(0), cv::Scalar::all(255));

    clock_gettime(CLOCK_MONOTONIC, &start_time);

    inference_yolov1_model(&rknn_app_ctx, &od_results);

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double inference_time_us = (end_time.tv_sec - start_time.tv_sec) * 1e6 + (end_time.tv_nsec - start_time.tv_nsec) / 1e3;
    printf("Inference Time: %.2f microseconds\n", inference_time_us);

    ret = release_yolov1_model(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release_yolov1_model fail! ret=%d\n", ret);
    }

    return 0;
}
