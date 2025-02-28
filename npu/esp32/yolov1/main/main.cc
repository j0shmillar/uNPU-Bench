#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include "esp_timer.h"

#include "main_functions.h"

extern "C" void app_main(void) 
{
  vTaskDelay(pdMS_TO_TICKS(50000));

  int64_t start_time = esp_timer_get_time();
  setup();
  int64_t end_time = esp_timer_get_time();
  printf("Init I/O time: %lld us\n", end_time - start_time);

  vTaskDelay(pdMS_TO_TICKS(50000));

  run();
}
