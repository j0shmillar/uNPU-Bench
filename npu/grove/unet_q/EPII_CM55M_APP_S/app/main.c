/**************************************************************************************************
 (C) COPYRIGHT, Himax Technologies, Inc. ALL RIGHTS RESERVED
 ------------------------------------------------------------------------
 File        : main.c
 Project     : EPII
 DATE        : 2021/04/01
 AUTHOR      : 902447
 BRIFE       : main function
 HISTORY     : Initial version - 2021/04/01 created by Jacky
 **************************************************************************************************/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>


#include "WE2_device.h"
#include "WE2_core.h"
#include "board.h"
#ifdef LIB_COMMON
#include "xprintf.h"
#include "console_io.h"
#endif
#include "pinmux_init.h"
#include "platform_driver_init.h"

#define CIS_XSHUT_SGPIO0

#ifdef HM_COMMON
#include "hx_drv_CIS_common.h"
#endif
#if defined(CIS_HM0360_MONO_REVB) || defined(CIS_HM0360_MONO_OSC_REVB) \
	|| defined(CIS_HM0360_BAYER_REVB) || defined(CIS_HM0360_BAYER_OSC_REVB) \
	||  defined(CIS_HM0360_MONO) || defined(CIS_HM0360_BAYER)
#include "hx_drv_hm0360.h"
#endif
#ifdef CIS_HM11B1
#include "hx_drv_hm11b1.h"
#endif
#if defined(CIS_HM01B0_MONO) || defined(CIS_HM01B0_BAYER)
#include "hx_drv_hm01b0.h"
#endif
#ifdef CIS_HM2140
#include "hx_drv_hm2140.h"
#endif
#ifdef CIS_XSHUT_SGPIO0
#define DEAULT_XHSUTDOWN_PIN    AON_GPIO2 
#else
#define DEAULT_XHSUTDOWN_PIN    AON_GPIO2 
#endif

#define CORE_CLOCK_HZ 100000000 // TODO pass as argument into app_main

#include "allon_sensor_tflm.h"

int main(void)
{
	board_init();

	SystemCoreClockUpdate(CORE_CLOCK_HZ);

	app_main();
	return 0;
}
