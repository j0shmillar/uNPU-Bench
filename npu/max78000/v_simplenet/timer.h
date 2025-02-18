#include "mxc.h"
#ifndef TIMER_H
#define TIMER_H


void timer_start(void);
volatile int timer_end(void);
void timer2_start(void);
volatile uint32_t timer2_end(void);
void timer0_start(void);
volatile int timer0_end(void);

#endif // TIMER_H