/**************************************************************************************************
* Copyright (C) 2019-2021 Maxim Integrated Products, Inc. All Rights Reserved.
*
* Maxim Integrated Products, Inc. Default Copyright Notice:
* https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
**************************************************************************************************/

/*
 * This header file was automatically @generated for the v_yolo_riscv network from a template.
 * Please do not edit; instead, edit the template and regenerate.
 */

#ifndef __CNN_H__
#define __CNN_H__

#include <stdint.h>
typedef int32_t q31_t;
typedef int16_t q15_t;

/* Return codes */
#define CNN_FAIL 0
#define CNN_OK 1

/*
  SUMMARY OF OPS
  Hardware: 22,910,976 ops (22,404,096 macc; 506,880 comp; 0 add; 0 mul; 0 bitwise)
    Layer 0: 4,128,768 ops (3,981,312 macc; 147,456 comp; 0 add; 0 mul; 0 bitwise)
    Layer 1: 5,492,736 ops (5,308,416 macc; 184,320 comp; 0 add; 0 mul; 0 bitwise)
    Layer 2: 1,373,184 ops (1,327,104 macc; 46,080 comp; 0 add; 0 mul; 0 bitwise)
    Layer 3: 2,672,640 ops (2,654,208 macc; 18,432 comp; 0 add; 0 mul; 0 bitwise)
    Layer 4: 304,128 ops (294,912 macc; 9,216 comp; 0 add; 0 mul; 0 bitwise)
    Layer 5: 2,672,640 ops (2,654,208 macc; 18,432 comp; 0 add; 0 mul; 0 bitwise)
    Layer 6: 304,128 ops (294,912 macc; 9,216 comp; 0 add; 0 mul; 0 bitwise)
    Layer 7: 2,672,640 ops (2,654,208 macc; 18,432 comp; 0 add; 0 mul; 0 bitwise)
    Layer 8: 94,464 ops (73,728 macc; 20,736 comp; 0 add; 0 mul; 0 bitwise)
    Layer 9: 668,160 ops (663,552 macc; 4,608 comp; 0 add; 0 mul; 0 bitwise)
    Layer 10: 76,032 ops (73,728 macc; 2,304 comp; 0 add; 0 mul; 0 bitwise)
    Layer 11: 668,160 ops (663,552 macc; 4,608 comp; 0 add; 0 mul; 0 bitwise)
    Layer 12: 76,032 ops (73,728 macc; 2,304 comp; 0 add; 0 mul; 0 bitwise)
    Layer 13: 668,160 ops (663,552 macc; 4,608 comp; 0 add; 0 mul; 0 bitwise)
    Layer 14: 76,032 ops (73,728 macc; 2,304 comp; 0 add; 0 mul; 0 bitwise)
    Layer 15: 668,160 ops (663,552 macc; 4,608 comp; 0 add; 0 mul; 0 bitwise)
    Layer 16: 152,064 ops (147,456 macc; 4,608 comp; 0 add; 0 mul; 0 bitwise)
    Layer 17: 76,032 ops (73,728 macc; 2,304 comp; 0 add; 0 mul; 0 bitwise)
    Layer 18: 39,168 ops (36,864 macc; 2,304 comp; 0 add; 0 mul; 0 bitwise)
    Layer 19: 27,648 ops (27,648 macc; 0 comp; 0 add; 0 mul; 0 bitwise)

  RESOURCE USAGE
  Weight memory: 42,352 bytes out of 442,368 bytes total (9.6%)
  Bias memory:   396 bytes out of 2,048 bytes total (19.3%)
*/

/* Number of outputs for this network */
// #define CNN_NUM_OUTPUTS 864
#define CNN_NUM_OUTPUTS 288

/* Use this timer to time the inference */
#define CNN_INFERENCE_TIMER MXC_TMR0

/* Port pin actions used to signal that processing is active */

#define CNN_START LED_On(1)
#define CNN_COMPLETE LED_Off(1)
#define SYS_START LED_On(0)
#define SYS_COMPLETE LED_Off(0)

/* Run software SoftMax on unloaded data */
void softmax_q17p14_q15(const q31_t * vec_in, const uint16_t dim_vec, q15_t * p_out);
/* Shift the input, then calculate SoftMax */
void softmax_shift_q17p14_q15(q31_t * vec_in, const uint16_t dim_vec, uint8_t in_shift, q15_t * p_out);

/* Stopwatch - holds the runtime when accelerator finishes */
extern volatile uint32_t cnn_time;

/* Custom memcopy routines used for weights and data */
void memcpy32(uint32_t *dst, const uint32_t *src, int n);
void memcpy32_const(uint32_t *dst, int n);

/* Enable clocks and power to accelerator, enable interrupt */
int cnn_enable(uint32_t clock_source, uint32_t clock_divider);

/* Disable clocks and power to accelerator */
int cnn_disable(void);

/* Perform minimum accelerator initialization so it can be configured */
int cnn_init(void);

/* Configure accelerator for the given network */
int cnn_configure(void);

/* Load accelerator weights */
int cnn_load_weights(void);

/* Verify accelerator weights (debug only) */
int cnn_verify_weights(void);

/* Load accelerator bias values (if needed) */
int cnn_load_bias(void);

/* Start accelerator processing */
int cnn_start(void);

/* Force stop accelerator */
int cnn_stop(void);

/* Continue accelerator after stop */
int cnn_continue(void);

/* Unload results from accelerator */
int cnn_unload(uint32_t *out_buf);

/* Turn on the boost circuit */
int cnn_boost_enable(mxc_gpio_regs_t *port, uint32_t pin);

/* Turn off the boost circuit */
int cnn_boost_disable(mxc_gpio_regs_t *port, uint32_t pin);

#endif // __CNN_H__
