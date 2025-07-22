/******************************************************************************
 *
 * Copyright (C) 2022-2023 Maxim Integrated Products, Inc. All Rights Reserved.
 * (now owned by Analog Devices, Inc.),
 * Copyright (C) 2023 Analog Devices, Inc. All Rights Reserved. This software
 * is proprietary to Analog Devices, Inc. and its licensors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 ******************************************************************************/

// cifar-100-residual
// Created using ai8xize.py --test-dir sdk/Examples/MAX78000/CNN --prefix cifar-100-residual --checkpoint-file trained/ai85-cifar100-residual-qat8-q.pth.tar --config-file networks/cifar100-ressimplenet.yaml --softmax --device MAX78000 --timer 0 --display-checkpoint --verbose --boost 2.5

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "mxc.h"
#include "cnn.h"
#include "sampledata.h"
#include "sampleoutput.h"


#include "timer.h"

#include "gcfr_regs.h"
#include "fcr_regs.h"
#include "sema_regs.h"

#include "tmr_revb.h"
#include <math.h>
#include <stdint.h>

#define TMR MXC_TMR2

volatile uint32_t cnn_time; // Stopwatch



void my_timer_initial()
{
    mxc_tmr_cfg_t cfg;
    cfg.pres = MXC_TMR_PRES_32;
    cfg.mode = MXC_TMR_MODE_ONESHOT;
    cfg.bitMode = MXC_TMR_BIT_MODE_32;
    cfg.clock = MXC_TMR_60M_CLK;
    cfg.cmp_cnt = 0xFFFFFFFF;
    cfg.pol = 0;
    MXC_TMR_Init(TMR, &cfg, false);
    MXC_TMR_Start(TMR);
}

void softmax_baseline(const float *vec_in, const uint16_t dim_vec, float *p_out)
{
    float sum = 0.0f;
    int16_t i;

    for (i = 0; i < dim_vec; i++)
    {
        p_out[i] = expf(vec_in[i]);
        sum += p_out[i];
    }

    for (i = 0; i < dim_vec; i++)
    {
        p_out[i] /= sum;
    }
}

void fail(void)
{
    printf("\n*** FAIL ***\n\n");
    while (1) {}
}

// 3-channel 32x32 data input (3072 bytes total / 1024 bytes per channel):
// HWC 32x32, channels 0 to 2
static const uint32_t input_60[] = SAMPLE_INPUT_60;

void load_input(void)
{
    // This function loads the sample data input -- replace with actual data

    memcpy32((uint32_t *)0x51018000, input_60, 1024);
}

// Expected output of layer 16 for cifar-100-residual given the sample input (known-answer test)
// Delete this function for production code
static const uint32_t sample_output[] = SAMPLE_OUTPUT;
int check_output(void)
{
    int i;
    uint32_t mask, len;
    volatile uint32_t *addr;
    const uint32_t *ptr = sample_output;

    while ((addr = (volatile uint32_t *)*ptr++) != 0) {
        mask = *ptr++;
        len = *ptr++;
        for (i = 0; i < len; i++)
            if ((*addr++ & mask) != *ptr++) {
                printf("Data mismatch (%d/%d) at address 0x%08x: Expected 0x%08x, read 0x%08x.\n",
                       i + 1, len, addr - 1, *(ptr - 1), *(addr - 1) & mask);
                return CNN_FAIL;
            }
    }

    return CNN_OK;
}

// // Classification layer:
// static int32_t ml_data[CNN_NUM_OUTPUTS];
// static q15_t ml_softmax[CNN_NUM_OUTPUTS];

// void softmax_layer(void)
// {
//     cnn_unload((uint32_t *)ml_data);
//     softmax_q17p14_q15((const q31_t *)ml_data, CNN_NUM_OUTPUTS, ml_softmax);
// }

// Customize the basic softmax
static float ml_data[CNN_NUM_OUTPUTS];
static float ml_softmax[CNN_NUM_OUTPUTS];

void softmax_layer(void)
{
    cnn_unload((uint32_t *)ml_data);
    softmax_baseline(ml_data, CNN_NUM_OUTPUTS, ml_softmax);
}

int main(void)
{

    MXC_ICC_Enable(MXC_ICC0); // Enable cache
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
    SystemCoreClockUpdate();


    // Configure Timer
    my_timer_initial();
    uint32_t prescale = (TMR->ctrl0 & MXC_F_TMR_REVB_CTRL0_CLKDIV_A) >> MXC_F_TMR_REVB_CTRL0_CLKDIV_A_POS;
    uint32_t timer_clock = MXC_TMR_RevB_GetClockSourceFreq((mxc_tmr_revb_regs_t*)TMR);
    printf("prescale: %lu, timer_clock: %lu\n", prescale, timer_clock);
    uint32_t start_ticks_init, end_ticks_init, elapsed_time_init, start_ticks_memory, end_ticks_memory, elapsed_time_memory, start_ticks_cnn, end_ticks_cnn, elapsed_time_cnn, start_ticks_cpu, end_ticks_cpu, elapsed_time_cpu;

    printf("Waiting...\n");

    // DO NOT DELETE THIS LINE:
    MXC_Delay(SEC(2)); // Let debugger interrupt if needed



    printf("\n*** CNN Inference Test ***\n");


    // ***************************** Initialization ***************************** //

    start_ticks_init = TMR->cnt;

    cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);
    cnn_boost_enable(MXC_GPIO2, MXC_GPIO_PIN_5); // Turn on the boost circuit
    cnn_init(); // Bring state machine into consistent state

    end_ticks_init = TMR->cnt;
    elapsed_time_init = (end_ticks_init - start_ticks_init) * (1 << (prescale & 0xF)) / (timer_clock / 1000000);



    // ***************************** Memory I/O ***************************** //
    start_ticks_memory = TMR->cnt;

    cnn_load_weights(); // Load kernels
    cnn_load_bias(); // Not used in this network
    cnn_configure(); // Configure state machine
    load_input(); // Load data input

    end_ticks_memory = TMR->cnt;
    elapsed_time_memory = (end_ticks_memory - start_ticks_memory) * (1 << (prescale & 0xF)) / (timer_clock / 1000000);


    // ***************************** Inference ***************************** //
    start_ticks_cnn = TMR->cnt;

    cnn_start(); // Start CNN processing
    while (cnn_time == 0)
        MXC_LP_EnterSleepMode(); // Wait for CNN
    cnn_boost_disable(MXC_GPIO2, MXC_GPIO_PIN_5); // Turn off the boost circuit

    end_ticks_cnn = TMR->cnt;
    elapsed_time_cnn = (end_ticks_cnn - start_ticks_cnn) * (1 << (prescale & 0xF)) / (timer_clock / 1000000);


    // ***************************** Post-Processing ***************************** //
    start_ticks_cpu = TMR->cnt;

    softmax_layer();

    end_ticks_cpu = TMR->cnt;
    elapsed_time_cpu = (end_ticks_cpu - start_ticks_cpu) * (1 << (prescale & 0xF)) / (timer_clock / 1000000);



    // ***************************** Finished ***************************** //
    cnn_disable(); // Shut down CNN clock, disable peripheral

    printf("\n*** PASS ***\n\n");

    printf("Initialization Time: %lu us\n", elapsed_time_init);
    printf("Memory I/O Time: %lu us\n", elapsed_time_memory);
    printf("Inference Time: %lu us\n", elapsed_time_cnn);
    printf("Post-Processing Time: %lu us\n", elapsed_time_cpu);

}

 

/*
  SUMMARY OF OPS
  Hardware: 18,636,416 ops (18,461,184 macc; 146,560 comp; 28,672 add; 0 mul; 0 bitwise)
    Layer 0: 458,752 ops (442,368 macc; 16,384 comp; 0 add; 0 mul; 0 bitwise)
    Layer 1: 2,969,600 ops (2,949,120 macc; 20,480 comp; 0 add; 0 mul; 0 bitwise)
    Layer 2: 0 ops (0 macc; 0 comp; 0 add; 0 mul; 0 bitwise)
    Layer 3: 3,706,880 ops (3,686,400 macc; 20,480 comp; 0 add; 0 mul; 0 bitwise)
    Layer 4: 3,727,360 ops (3,686,400 macc; 20,480 comp; 20,480 add; 0 mul; 0 bitwise)
    Layer 5: 947,200 ops (921,600 macc; 25,600 comp; 0 add; 0 mul; 0 bitwise)
    Layer 6: 0 ops (0 macc; 0 comp; 0 add; 0 mul; 0 bitwise)
    Layer 7: 926,720 ops (921,600 macc; 5,120 comp; 0 add; 0 mul; 0 bitwise)
    Layer 8: 2,043,904 ops (2,027,520 macc; 11,264 comp; 5,120 add; 0 mul; 0 bitwise)
    Layer 9: 1,230,848 ops (1,216,512 macc; 14,336 comp; 0 add; 0 mul; 0 bitwise)
    Layer 10: 0 ops (0 macc; 0 comp; 0 add; 0 mul; 0 bitwise)
    Layer 11: 1,330,176 ops (1,327,104 macc; 3,072 comp; 0 add; 0 mul; 0 bitwise)
    Layer 12: 671,232 ops (663,552 macc; 4,608 comp; 3,072 add; 0 mul; 0 bitwise)
    Layer 13: 200,192 ops (196,608 macc; 3,584 comp; 0 add; 0 mul; 0 bitwise)
    Layer 14: 262,656 ops (262,144 macc; 512 comp; 0 add; 0 mul; 0 bitwise)
    Layer 15: 148,096 ops (147,456 macc; 640 comp; 0 add; 0 mul; 0 bitwise)
    Layer 16: 12,800 ops (12,800 macc; 0 comp; 0 add; 0 mul; 0 bitwise)

  RESOURCE USAGE
  Weight memory: 381,792 bytes out of 442,368 bytes total (86%)
  Bias memory:   0 bytes out of 2,048 bytes total (0%)
*/
