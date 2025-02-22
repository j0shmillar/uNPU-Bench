#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "WE2_device.h"
#include "WE2_core.h"
#include "board.h"
#include "pinmux_init.h"
#include "platform_driver_init.h"


#ifdef HM_COMMON
#include "hx_drv_CIS_common.h"
#endif

#define DEAULT_XHSUTDOWN_PIN    AON_GPIO2

#include "allon_sensor_tflm.h"

int main(void)
{
	board_init();
	app_main();
	return 0;
}




