#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "model.h"
#include "common.h"

#define _POSIX_C_SOURCE 199309L
#include <time.h>

#define Q15_MAX_VALUE   32767
#define Q15_MIN_VALUE   -32768

typedef int32_t q31_t;
typedef int16_t q15_t;

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

int init_model(const char *model_path, rknn_app_context_t *app_ctx)
{
    int ret;
    int model_len = 0;
    char *model;
    rknn_context ctx = 0;

    ret = rknn_init(&ctx, (char *)model_path, 0, 0, NULL);
    if (ret < 0)
    {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }

    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    input_attrs[0].type = RKNN_TENSOR_UINT8;
    input_attrs[0].fmt = RKNN_TENSOR_NHWC;
    app_ctx->input_mems[0] = rknn_create_mem(ctx, input_attrs[0].size_with_stride);

    ret = rknn_set_io_mem(ctx, app_ctx->input_mems[0], &input_attrs[0]);
    if (ret < 0) {
        printf("input_mems rknn_set_io_mem fail! ret=%d\n", ret);
        return -1;
    }

    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        app_ctx->output_mems[i] = rknn_create_mem(ctx, output_attrs[i].size_with_stride);
        ret = rknn_set_io_mem(ctx, app_ctx->output_mems[i], &output_attrs[i]);
        if (ret < 0) {
            printf("output_mems rknn_set_io_mem fail! ret=%d\n", ret);
            return -1;
        }
    }

    app_ctx->rknn_ctx = ctx;

    if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC)
    {
        app_ctx->is_quant = true;
    }
    else
    {
        app_ctx->is_quant = false;
    }

    app_ctx->io_num = io_num;
    app_ctx->input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx->output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) 
    {
        printf("model is NCHW input fmt\n");
        app_ctx->model_channel = input_attrs[0].dims[1];
        app_ctx->model_height  = input_attrs[0].dims[2];
        app_ctx->model_width   = input_attrs[0].dims[3];
    } else 
    {
        printf("model is NHWC input fmt\n");
        app_ctx->model_height  = input_attrs[0].dims[1];
        app_ctx->model_width   = input_attrs[0].dims[2];
        app_ctx->model_channel = input_attrs[0].dims[3];
    } 

    printf("model input height=%d, width=%d, channel=%d\n",
           app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel);

    return 0;
}

int release_model(rknn_app_context_t *app_ctx)
{
    if (app_ctx->rknn_ctx != 0)
    {
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    if (app_ctx->input_attrs != NULL)
    {
        free(app_ctx->input_attrs);
        app_ctx->input_attrs = NULL;
    }
    if (app_ctx->output_attrs != NULL)
    {
        free(app_ctx->output_attrs);
        app_ctx->output_attrs = NULL;
    }
    for (int i = 0; i < app_ctx->io_num.n_input; i++) {
        if (app_ctx->input_mems[i] != NULL) {
            rknn_destroy_mem(app_ctx->rknn_ctx, app_ctx->input_mems[i]);
            free(app_ctx->input_mems[i]);
        }
    }
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        if (app_ctx->output_mems[i] != NULL) {
            rknn_destroy_mem(app_ctx->rknn_ctx, app_ctx->output_mems[i]);
            free(app_ctx->output_mems[i]);
        }
    }
    return 0;
}

int inference_model(rknn_app_context_t *app_ctx)
{
    struct timespec start_time, end_time;

    int ret;

    clock_gettime(CLOCK_MONOTONIC, &start_time);
    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double inference_time_us = (end_time.tv_sec - start_time.tv_sec) * 1e6 + (end_time.tv_nsec - start_time.tv_nsec) / 1e3;
    printf("Inference time: %.2f microseconds\n", inference_time_us);

    float processed_output[768];
    size_t output_size = 768 * sizeof(float);
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    if (output_size > app_ctx->output_attrs[0].size) {
        printf("Error: Output buffer size exceeds expected size!\n");
        return -1;
    }
    memcpy(processed_output, app_ctx->output_mems[0]->virt_addr, output_size);
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double memory_time_us = (end_time.tv_sec - start_time.tv_sec) * 1e6 + (end_time.tv_nsec - start_time.tv_nsec) / 1e3;
    printf("Memory I/O time: %.2f microseconds\n", memory_time_us);

    return ret;
}
