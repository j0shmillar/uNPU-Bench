#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "model.h"
#include "common.h"

#define GRID_SIZE 12
#define NUM_CLASSES 2  // Based on the output tensor shape shown
#define NUM_CONFIDENCE 10 // Based on the slice operations shown
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

void process_output(const float* output_array, float* final_output, int grid_size) {
    const int confidence_stride = grid_size * grid_size * 2;  
    
    for (int y = 0; y < grid_size; y++) {
        for (int x = 0; x < grid_size; x++) {
	    printf("here3\n");
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

int inference_model(rknn_app_context_t *app_ctx, object_detect_result_list *od_results)
{
    int ret;
    const float nms_threshold = NMS_THRESH;   
    const float box_conf_threshold = BOX_THRESH;
   
    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }

    float processed_output[GRID_SIZE * GRID_SIZE * OUTPUT_STRIDE];
    detection_t detections[GRID_SIZE * GRID_SIZE * 2]; 
    int num_detections;

    process_output((float*)app_ctx->output_mems[0], processed_output, GRID_SIZE);

    float confidence_threshold = 0.5f;
    extract_detections(processed_output, detections, &num_detections, confidence_threshold, GRID_SIZE);

    printf("\nDetections found: %d\n", num_detections);
    for (int i = 0; i < num_detections; i++) {
        printf("Detection %d: (x=%.3f, y=%.3f) conf=%.3f class=%d score=%.3f\n",
            i,
            detections[i].x,
            detections[i].y,
            detections[i].confidence,
            detections[i].class_id,
            detections[i].class_probs[detections[i].class_id]);
    }

    return ret;
}
