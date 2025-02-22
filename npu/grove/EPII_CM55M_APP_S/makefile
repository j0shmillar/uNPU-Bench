EPII_ROOT = .

APPL ?= EPII_CM55M

OLEVEL ?= O2
DEBUG ?= 1
BOARD ?= epii_evb
BD_VER ?= 10
CORTEX_M ?= 55
#BIG, SMALL
EPII_CORE ?= BIG

IC_PACKAGE_SEL = WLCSP65

IC_VER ?= 30

CoreVolt ?= 0P9V

TOOLCHAIN ?= gnu

ARM_TOOLPATH ?=/Users/joshmillar/Desktop/phd/mcu-nn-eval/npu/grove/yolov1_size/arm-gnu-toolchain-13.2.Rel1-x86_64-arm-none-eabi
GNU_TOOLPATH ?=
#ARM_TOOLPATH ?= C:/Program Files/Arm/Development Studio 2020.1/sw/ARMCompiler6.15/bin
#GNU_TOOLPATH ?= D:/Project/WE-II/CMSIS/cmsis_tool/gcc-arm-none-eabi-10-2020-q4-major-win32/gcc-arm-none-eabi-10-2020-q4-major/bin

OS_SEL ?= freertos

MPU ?= y

TRUSTZONE ?= y
TRUSTZONE_TYPE ?= security
TRUSTZONE_FW_TYPE ?= 0
NONTRUSTZONE_SRC_DIR ?= D:\ARMDS_WORKSPACE\EPII_CM55M_APP_NS



CIS_SEL =

EPII_USECASE_SEL = drv_onecore_cm55m_s_only


EPII_NSC_SEL = cm55m_nsc_cfg

LIB_CMSIS_NN_ENALBE = 1
LIB_CMSIS_NN_VERSION = 7_0_0

LIB_CV_PREBUILT = 0

LIB_AUDIOALGO_PREBUILT = 0

CMSIS_DRIVERS_USECASE_SEL =

SUPPORT_SEC_LIB ?= y

CUSTOMER ?= seeed

APP_TYPE = allon_sensor_tflm
 
override EPII_ROOT := $(strip $(subst \,/,$(EPII_ROOT)))
include $(EPII_ROOT)/options/options.mk
