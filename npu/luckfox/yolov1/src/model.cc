#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "model.h"
#include "common.h"

#define GRID_SIZE 12
#define NUM_CLASSES 2  
#define NUM_CONFIDENCE 10 
#define OUTPUT_STRIDE (NUM_CLASSES + NUM_CONFIDENCE)

#define OUT_H 12
#define OUT_W 12

static float processed_output[GRID_SIZE * GRID_SIZE * OUTPUT_STRIDE];

#define Q15_MAX_VALUE   32767
#define Q15_MIN_VALUE   -32768

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

int inference_model(rknn_app_context_t *app_ctx, object_detect_result_list *od_results)
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

    size_t output_size = GRID_SIZE * GRID_SIZE * OUTPUT_STRIDE * sizeof(float);
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    if (output_size > app_ctx->output_attrs[0].size) {
        printf("Error: Output buffer size exceeds expected size!\n");
        return -1;
    }
    memcpy(processed_output, app_ctx->output_mems[0]->virt_addr, output_size);
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double memory_time_us = (end_time.tv_sec - start_time.tv_sec) * 1e6 + (end_time.tv_nsec - start_time.tv_nsec) / 1e3;
    printf("Memory I/O time: %.2f microseconds\n", memory_time_us);

    q31_t**** array1 = generateArray(2);
    q31_t**** array2 = generateArray(10);
    
    q31_t* class_output = &array1[0][0][0][0]; 
    q31_t* obj_confidence = &array2[0][0][0][0];  

    q15_t softmax_output[OUT_H * OUT_W * NUM_CLASSES];
    q15_t sigmoid_output[OUT_H * OUT_W * NUM_CONFIDENCE];

    clock_gettime(CLOCK_MONOTONIC, &start_time);
    for (int i = 0; i < OUT_H * OUT_W; i++) {
        softmax_q17p14_q15(&class_output[i * NUM_CLASSES], NUM_CLASSES, &softmax_output[i * NUM_CLASSES]);
    }

    for (int i = 0; i < OUT_H * OUT_W; i++) {
        sigmoid_q15(&obj_confidence[i * NUM_CLASSES], NUM_CLASSES, &sigmoid_output[i * NUM_CLASSES]);
    }
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double postproc_time_us = (end_time.tv_sec - start_time.tv_sec) * 1e6 + (end_time.tv_nsec - start_time.tv_nsec) / 1e3;
    printf("Post-proc time: %.2f microseconds\n", postproc_time_us);

    return ret;
}
