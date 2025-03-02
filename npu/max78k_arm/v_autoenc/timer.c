
#include "mxc.h"
#define CUSTOM_TIMER MXC_TMR1
#define CUSTOM_TIMER2 MXC_TMR2
#define CUSTOM_TIMER0 MXC_TMR0


void timer_start(void)
{
  MXC_TMR_SW_Start(CUSTOM_TIMER);
}

volatile int timer_end()
{
  return MXC_TMR_SW_Stop(CUSTOM_TIMER);
}


// Function to start a timer
void timer0_start(void)
{
    MXC_TMR_SW_Start(CUSTOM_TIMER0);
}

// Function to stop the timer and return elapsed time in microseconds
volatile int timer0_end(void)
{
    return MXC_TMR_SW_Stop(CUSTOM_TIMER0);
}


// 230922: Note that using multiple timers at once resulted in unexpeted addtional latency, e.g., 2000~8000us
void timer2_start(void)
{
  MXC_TMR_SW_Start(CUSTOM_TIMER2);
}

volatile uint32_t timer2_end()
{
  return MXC_TMR_SW_Stop(CUSTOM_TIMER2);
}
