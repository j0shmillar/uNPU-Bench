#ifndef APP_SCENARIO_ALLON_SENSOR_TFLM_COMMON_CONFIG_H_
#define APP_SCENARIO_ALLON_SENSOR_TFLM_COMMON_CONFIG_H_

/** model location:
 *	0: model file is a c file which will locate to memory.
 *
 *	1: model file will off-line burn to dedicated location in flash,
 *		use flash memory mapped address to load model.
 *		e.g. model data is pre-burn to flash address: 0x180000
 * **/
#define FLASH_XIP_MODEL 0
#define MEM_FREE_POS		(BOOT2NDLOADER_BASE) ////0x3401F000

#endif /* APP_SCENARIO_ALLON_SENSOR_TFLM_COMMON_CONFIG_H_ */
