#include <stdio.h>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "powermode_export.h"

#define WE2_CHIP_VERSION_C		0x8538000c
#define FRAME_CHECK_DEBUG		1
#ifdef TRUSTZONE_SEC
#ifdef FREERTOS
/* Trustzone config. */
//
/* FreeRTOS includes. */
//#include "secure_port_macros.h"
#else
#if (__ARM_FEATURE_CMSE & 1) == 0
#error "Need ARMv8-M security extensions"
#elif (__ARM_FEATURE_CMSE & 2) == 0
#error "Compile with --cmse"
#endif
#include "arm_cmse.h"
//#include "veneer_table.h"
//
#endif
#endif

#include "WE2_device.h"
#include "cvapp.h"
#include "spi_master_protocol.h"
#include "hx_drv_spi.h"
#include "spi_eeprom_comm.h"
#include "board.h"
#include "xprintf.h"
#include "allon_sensor_tflm.h"
#include "board.h"
#include "WE2_core.h"
#include "hx_drv_scu.h"
#include "hx_drv_swreg_aon.h"
#ifdef IP_sensorctrl
#include "hx_drv_sensorctrl.h"
#endif
#ifdef IP_xdma
#include "hx_drv_xdma.h"
#include "sensor_dp_lib.h"
#endif
#ifdef IP_cdm
#include "hx_drv_cdm.h"
#endif
#ifdef IP_gpio
#include "hx_drv_gpio.h"
#endif
#include "hx_drv_pmu_export.h"
#include "hx_drv_pmu.h"
#include "powermode.h"
#include "math.h"
#include "BITOPS.h"

#include "common_config.h"
#include "model_data.h"

#ifdef EPII_FPGA
#define DBG_APP_LOG             (1)
#else
#define DBG_APP_LOG             (1)
#endif
#if DBG_APP_LOG
    #define dbg_app_log(fmt, ...)       xprintf(fmt, ##__VA_ARGS__)
#else
    #define dbg_app_log(fmt, ...)
#endif

#define MAX_STRING  100
#define CHAR_BUFF_SIZE 50 
#define DEBUG_SPIMST_SENDPICS		(0x01)
#define CPU_FREQ_MHZ 100

#define INIT_DWT()      (DWT->CTRL |= 1) 
#define RESET_DWT()     (DWT->CYCCNT = 0) 
#define GET_DWT()       (DWT->CYCCNT)   

#include "hx_drv_scu.h"
#include "hx_drv_pmu.h"
#include "WE2_device.h"

void set_clock(uint32_t speed)
{
	hx_drv_swreg_aon_set_pllfreq(speed);
	SystemCoreClock = speed;
    xprintf("Clock set to 100 MHz\n");
}

static char * _float_to_char(float x, char *p) {
    char *s = p + CHAR_BUFF_SIZE - 1;  
    *s = '\0';  

    uint16_t decimals;
    int units;

    if (x < 0) { 
        decimals = (int)(x * -100) % 100; 
        units = (int)(-1 * x);
    } else { 
        decimals = (int)(x * 100) % 100;
        units = (int)x;
    }

    *--s = (decimals % 10) + '0';
    decimals /= 10; 
    *--s = (decimals % 10) + '0';
    *--s = '.';

    if (units == 0) {
        *--s = '0';
    } else {
        while (units > 0) {
            *--s = (units % 10) + '0';
            units /= 10;
        }
    }

    if (x < 0) *--s = '-';  

    return s;
}


void enable_dwt()
{
    if (!(CoreDebug->DEMCR & CoreDebug_DEMCR_TRCENA_Msk)) {
        CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk; 
        DWT->CYCCNT = 0; 
        DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk; 
    }
}

static void dp_app_cv_eventhdl_cb()
{
	cv_run();
}

void app_start_state()
{
	dp_app_cv_eventhdl_cb();
}

void delay_cycles(uint32_t cycles) {
    while (cycles--) {
        __NOP();
    }
}

int app_main(void) {

    set_clock(100000000);
    printf("clock speed: %lu Hz\n", SystemCoreClock);

    enable_dwt(); 
    RESET_DWT(); 

    delay_cycles(500000000); 

    uint32_t start = GET_DWT();
    if(cv_init(true, true) < 0) {
        xprintf("cv init fail\n");
        return -1;
    }
    uint32_t end = GET_DWT();
    uint32_t cycles = end - start;
    float time_us = (float)cycles / CPU_FREQ_MHZ; 
    char time_str[CHAR_BUFF_SIZE];  
    char *time_ptr = _float_to_char(time_us, time_str);  
    xprintf("Init time: %s us\n", time_ptr);  

    delay_cycles(500000000); 

	app_start_state();
	
    return 0;
}