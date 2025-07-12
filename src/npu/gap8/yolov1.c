#include "pmsis.h"
#include "yolov1.h"
#include "yolov1Kernels.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>  

#define pmsis_exit(n) exit(n)

#ifndef STACK_SIZE
#define STACK_SIZE      1024
#endif

AT_HYPERFLASH_FS_EXT_ADDR_TYPE yolov1_L3_Flash = 0;

L2_MEM short int *ResOut;
L2_MEM unsigned char *Img_In;

#define AT_INPUT_SIZE (AT_INPUT_WIDTH*AT_INPUT_HEIGHT*AT_INPUT_COLORS)
#define AT_INPUT_SIZE_BYTES (AT_INPUT_SIZE*sizeof(char))

#define GRID_SIZE 12
#define N_CLASSES 2  
#define NUM_CONFIDENCE 10 

#define OUT_H 12
#define OUT_W 12

#define Q15_MAX_VALUE   32767
#define Q15_MIN_VALUE   -32768

typedef int32_t q31_t;
typedef int16_t q15_t;

static unsigned int seed = 12345678;  

unsigned int custom_rand(void) {
    seed = seed * 1103515245 + 12345;
    return (seed / 65536) % 32768;
}

static void generate_random_image()
{
    for (int i = 0; i < AT_INPUT_SIZE; i++)
    {
        Img_In[i] = custom_rand() % 256; 
    }
    printf("Random image generated.\n");
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

static void cluster()
{
    pi_perf_conf(1 << PI_PERF_CYCLES | 1 << PI_PERF_ACTIVE_CYCLES);
    pi_perf_reset();
    pi_perf_start(); 
    yolov1CNN(Img_In, ResOut);
    pi_perf_stop();
    // uint32_t cycles = pi_perf_read(PI_PERF_ACTIVE_CYCLES);
    uint32_t cycles = pi_perf_read(PI_PERF_CYCLES); 
    uint32_t clock_frequency = pi_freq_get(PI_FREQ_DOMAIN_CL);  
    double elapsed_time_us = ((double)cycles / clock_frequency) * 1000000;
    printf("Infer (+ Memory I/O) time: %lf us\n", elapsed_time_us);

    // pi_perf_reset();
    // pi_perf_start();
    // short int ResOutCopy[N_CLASSES];
    // memcpy(ResOutCopy, ResOut, N_CLASSES * sizeof(short int));
    // pi_perf_stop();
    // cycles = pi_perf_read(PI_PERF_ACTIVE_CYCLES); 
    // elapsed_time_us = ((double)cycles / clock_frequency) * 1000000;  
    // printf("Memory I/O time: %lf us\n", elapsed_time_us);
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

int test_yolov1(void)
{   
    pi_perf_conf(1 << PI_PERF_CYCLES | 1 << PI_PERF_ACTIVE_CYCLES);
    pi_perf_reset();

    pi_freq_set(PI_FREQ_DOMAIN_FC, 100*1000*1000);
    pi_freq_set(PI_FREQ_DOMAIN_CL, 100*1000*1000);

    uint32_t clock_frequency = pi_freq_get(PI_FREQ_DOMAIN_CL);  
    printf("clock freq: %u Hz\n", clock_frequency);

    pi_perf_start();
    Img_In = (unsigned char *) AT_L2_ALLOC(0, AT_INPUT_SIZE_BYTES);
    if(Img_In == NULL) {
        printf("Image buffer alloc Error!\n");
        pmsis_exit(-1);
    }
    generate_random_image();
    ResOut = (short int *) AT_L2_ALLOC(0, 10 * sizeof(short int));
    if (ResOut == NULL) {
        printf("Failed to allocate Memory for Result (%d bytes)\n", 10 * sizeof(short int));
        pmsis_exit(-3);
    }
    pi_perf_stop();
    uint32_t cycles = pi_perf_read(PI_PERF_CYCLES);
    double elapsed_time_us = ((double)cycles / clock_frequency) * 1000000; 
    printf("Memory I/O time: %lf us\n", elapsed_time_us);

    struct pi_device cluster_dev;
    struct pi_cluster_conf cl_conf;
    cl_conf.id = 0;
    cl_conf.cc_stack_size = STACK_SIZE;
    pi_perf_reset();
    pi_perf_start();
    pi_cluster_conf_init(&cl_conf);
    pi_open_from_conf(&cluster_dev, (void *) &cl_conf);
    if (pi_cluster_open(&cluster_dev)) {
        printf("Cluster open failed !\n");
        pmsis_exit(-4);
    }
    int construct_err = yolov1CNN_Construct(0);
    if (construct_err) {
        printf("Graph constructor exited with error: %d\n", construct_err);
        pmsis_exit(-5);
    }
    pi_perf_stop();
    cycles = pi_perf_read(PI_PERF_CYCLES);
    elapsed_time_us = ((double)cycles / clock_frequency) * 1000000; 
    printf("Init time: %lf us\n", elapsed_time_us);

    printf("Call cluster\n");
    struct pi_cluster_task task;
    pi_cluster_task(&task, cluster, NULL);
    pi_cluster_task_stacks(&task, NULL, STACK_SIZE);
    pi_cluster_send_task(&cluster_dev, &task);

    printf("Deallocate everything\n");
    yolov1CNN_Destruct(0);
    pi_cluster_close(&cluster_dev);

    q31_t**** array1 = generateArray(2);
    q31_t**** array2 = generateArray(10);
    q31_t* class_output = &array1[0][0][0][0]; 
    q15_t softmax_output[OUT_H * OUT_W * N_CLASSES];

    pi_perf_reset();
    pi_perf_start();
    for (int i = 0; i < OUT_H * OUT_W; i++) {
        softmax_q17p14_q15(&class_output[i * N_CLASSES], N_CLASSES, &softmax_output[i * N_CLASSES]);
    }
    q15_t *sigmoid_output;
    q31_t *obj_confidence;
    sigmoid_output = (q15_t *) AT_L2_ALLOC(0, OUT_H * OUT_W * NUM_CONFIDENCE * sizeof(q15_t));
    if (!sigmoid_output) {
        printf("Failed to allocate sigmoid_output\n");
        pmsis_exit(-1);
    }
    obj_confidence = (q31_t *) AT_L2_ALLOC(0, OUT_H * OUT_W * NUM_CONFIDENCE * sizeof(q31_t));
    if (!obj_confidence) {
        printf("Failed to allocate obj_confidence\n");
        pmsis_exit(-1);
    }
    memset(sigmoid_output, 0, OUT_H * OUT_W * NUM_CONFIDENCE * sizeof(q15_t));
    memset(obj_confidence, 0, OUT_H * OUT_W * NUM_CONFIDENCE * sizeof(q31_t));
    for (int i = 0; i < OUT_H * OUT_W; i++) {
        sigmoid_q15(&obj_confidence[i * NUM_CONFIDENCE], NUM_CONFIDENCE, &sigmoid_output[i * NUM_CONFIDENCE]);
    }
    pi_perf_stop();
    cycles = pi_perf_read(PI_PERF_CYCLES);
    elapsed_time_us = ((double)cycles / clock_frequency) * 1000000; 
    printf("Post-proc time: %lf us\n", elapsed_time_us);

    AT_L2_FREE(0, sigmoid_output, OUT_H * OUT_W * NUM_CONFIDENCE * sizeof(q15_t));
    AT_L2_FREE(0, obj_confidence, OUT_H * OUT_W * NUM_CONFIDENCE * sizeof(q31_t));

    AT_L2_FREE(0, Img_In, AT_INPUT_SIZE_BYTES);
    AT_L2_FREE(0, ResOut, 10 * sizeof(short int));

    pmsis_exit(0);
    return 0;
}

int main()
{
    printf("\n\n\t *** NNTOOL yolov1 ***\n\n");
    return pmsis_kickoff((void *) test_yolov1);
}
