#include <stdio.h>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "powermode_export.h"

#define FRAME_CHECK_DEBUG 1
#ifdef TRUSTZONE_SEC
#ifdef FREERTOS
/* Trustzone config */
/* FreeRTOS includes */
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
#include "BITOPS.h"

#include "common_config.h"
#include "yolo.h"

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
#define DEBUG_SPIMST_SENDPICS		(0x01) //0x00: off/ 0x01: JPEG/0x02: YUV422/0x03: YUV420/0x04: YUV400/0x05: RGB
#define SPI_SEN_PIC_CLK				(10000000)


// flash image position
/*volatile*/ uint32_t g_flash_record_start_pos = 0;
/*volatile*/ uint32_t g_flash_image_cur_pos = 0;
/*volatile*/ uint32_t g_flash_length_cur_pos = 0;
/*volatile*/ uint32_t g_idle_time = 0;
/*volatile*/ uint32_t g_img_data = 0;

static uint8_t 	g_xdma_abnormal, g_md_detect, g_cdm_fifoerror, g_wdt1_timeout, g_wdt2_timeout,g_wdt3_timeout;
static uint8_t 	g_hxautoi2c_error, g_inp1bitparer_abnormal;
static uint32_t g_dp_event;
static uint8_t 	g_frame_ready;
static uint32_t g_cur_jpegenc_frame;
static uint8_t 	g_time;
static uint8_t g_spi_master_initial_status;
/*volatile*/ uint32_t jpeg_addr, jpeg_sz;

void app_start_state(APP_STATE_E state);


static void dp_var_int()
{
	g_xdma_abnormal = 0;
	g_md_detect = 0;
	g_cdm_fifoerror = 0;
	g_wdt1_timeout = 0;
	g_wdt2_timeout = 0;
	g_wdt3_timeout = 0;
	g_inp1bitparer_abnormal = 0;
	g_dp_event = 0;
	g_frame_ready = 0;
	g_time = 0;
	g_cur_jpegenc_frame = 0;
	g_hxautoi2c_error = 0;
	g_spi_master_initial_status = 0;
}

int dp_app_cv_eventhdl_cb()
{
	int res = cv_run();
	return res;
}

void app_start_state(APP_STATE_E state)
{
	if(state == APP_STATE_ALLON) 
	{

        dp_var_int();

        //if wdma is zero when not init yet, this step retrieves wdma address (must)
        if(cv_run())
        {
        	xprintf("\r\ninference success\r\n");
        }
	}
}


int app_main(void) 
{

    if(cv_init(true, true)<0) 
	{
    	xprintf("cv init fail\n");
    	return -1;
    }

    app_start_state(APP_STATE_ALLON);

	return 0;
}
