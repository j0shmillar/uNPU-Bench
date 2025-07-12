#include "board_init.h"
#include "fsl_debug_console.h"
#include "model.h"
#include "timer.h"
#include <stdio.h>
#include "math.h"

extern void lower_active_power();

int main(void)
{
    BOARD_Init();
    TIMER_Init();
    
    lower_active_power();

    run();

    while(1)
    {

    }
}
