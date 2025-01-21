################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../drivers/fsl_clock.c \
../drivers/fsl_common.c \
../drivers/fsl_common_arm.c \
../drivers/fsl_ctimer.c \
../drivers/fsl_edma.c \
../drivers/fsl_edma_soc.c \
../drivers/fsl_flexio.c \
../drivers/fsl_flexio_mculcd.c \
../drivers/fsl_flexio_mculcd_edma.c \
../drivers/fsl_gpio.c \
../drivers/fsl_inputmux.c \
../drivers/fsl_lpflexcomm.c \
../drivers/fsl_lpi2c.c \
../drivers/fsl_lpuart.c \
../drivers/fsl_reset.c \
../drivers/fsl_sctimer.c \
../drivers/fsl_smartdma.c \
../drivers/fsl_spc.c 

C_DEPS += \
./drivers/fsl_clock.d \
./drivers/fsl_common.d \
./drivers/fsl_common_arm.d \
./drivers/fsl_ctimer.d \
./drivers/fsl_edma.d \
./drivers/fsl_edma_soc.d \
./drivers/fsl_flexio.d \
./drivers/fsl_flexio_mculcd.d \
./drivers/fsl_flexio_mculcd_edma.d \
./drivers/fsl_gpio.d \
./drivers/fsl_inputmux.d \
./drivers/fsl_lpflexcomm.d \
./drivers/fsl_lpi2c.d \
./drivers/fsl_lpuart.d \
./drivers/fsl_reset.d \
./drivers/fsl_sctimer.d \
./drivers/fsl_smartdma.d \
./drivers/fsl_spc.d 

OBJS += \
./drivers/fsl_clock.o \
./drivers/fsl_common.o \
./drivers/fsl_common_arm.o \
./drivers/fsl_ctimer.o \
./drivers/fsl_edma.o \
./drivers/fsl_edma_soc.o \
./drivers/fsl_flexio.o \
./drivers/fsl_flexio_mculcd.o \
./drivers/fsl_flexio_mculcd_edma.o \
./drivers/fsl_gpio.o \
./drivers/fsl_inputmux.o \
./drivers/fsl_lpflexcomm.o \
./drivers/fsl_lpi2c.o \
./drivers/fsl_lpuart.o \
./drivers/fsl_reset.o \
./drivers/fsl_sctimer.o \
./drivers/fsl_smartdma.o \
./drivers/fsl_spc.o 


# Each subdirectory must supply rules for building sources it contributes
drivers/%.o: ../drivers/%.c drivers/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: MCU C Compiler'
	arm-none-eabi-gcc -std=gnu99 -D__NEWLIB__ -DPRINTF_FLOAT_ENABLE=1 -DCPU_MCXN947VDF -DCPU_MCXN947VDF_cm33 -DCPU_MCXN947VDF_cm33_core0 -DSDK_DEBUGCONSOLE_UART -DARM_MATH_CM33 -D__FPU_PRESENT=1 -DTF_LITE_STATIC_MEMORY -DMCUXPRESSO_SDK -DSDK_DEBUGCONSOLE=1 -DCR_INTEGER_PRINTF -D__MCUXPRESSO -D__USE_CMSIS -DDEBUG -DLCD_IMPL_FLEXIO=1 -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/source" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/utilities" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/eiq/tensorflow-lite" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/eiq/tensorflow-lite/third_party/flatbuffers/include" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/eiq/tensorflow-lite/third_party/gemmlowp" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/component/lists" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/component/uart" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/drivers" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/device" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/startup" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/eiq/tensorflow-lite/tensorflow/lite/micro/kernels/neutron" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/eiq/tensorflow-lite/third_party/ruy" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/CMSIS" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/eiq/tensorflow-lite/third_party/neutron/common/include" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/eiq/tensorflow-lite/third_party/neutron/driver/include" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/source/image" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/source/model" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/board" -O3 -fno-common -g3 -gdwarf-4 -Wall -Wno-strict-aliasing -mcpu=cortex-m33 -c -ffunction-sections -fdata-sections -fmacro-prefix-map="$(<D)/"= -mcpu=cortex-m33 -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb -D__NEWLIB__ -fstack-usage -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.o)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


clean: clean-drivers

clean-drivers:
	-$(RM) ./drivers/fsl_clock.d ./drivers/fsl_clock.o ./drivers/fsl_common.d ./drivers/fsl_common.o ./drivers/fsl_common_arm.d ./drivers/fsl_common_arm.o ./drivers/fsl_ctimer.d ./drivers/fsl_ctimer.o ./drivers/fsl_edma.d ./drivers/fsl_edma.o ./drivers/fsl_edma_soc.d ./drivers/fsl_edma_soc.o ./drivers/fsl_flexio.d ./drivers/fsl_flexio.o ./drivers/fsl_flexio_mculcd.d ./drivers/fsl_flexio_mculcd.o ./drivers/fsl_flexio_mculcd_edma.d ./drivers/fsl_flexio_mculcd_edma.o ./drivers/fsl_gpio.d ./drivers/fsl_gpio.o ./drivers/fsl_inputmux.d ./drivers/fsl_inputmux.o ./drivers/fsl_lpflexcomm.d ./drivers/fsl_lpflexcomm.o ./drivers/fsl_lpi2c.d ./drivers/fsl_lpi2c.o ./drivers/fsl_lpuart.d ./drivers/fsl_lpuart.o ./drivers/fsl_reset.d ./drivers/fsl_reset.o ./drivers/fsl_sctimer.d ./drivers/fsl_sctimer.o ./drivers/fsl_smartdma.d ./drivers/fsl_smartdma.o ./drivers/fsl_spc.d ./drivers/fsl_spc.o

.PHONY: clean-drivers

