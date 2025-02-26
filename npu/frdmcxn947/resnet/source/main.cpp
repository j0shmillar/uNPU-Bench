#include "board_init.h"
#include "fsl_debug_console.h"
#include "model.h"
#include "timer.h"
#include <stdio.h>
#include "math.h"

int main(void)
{
    BOARD_Init();
    TIMER_Init();

    run();

    while(1)
    {

    }
}
