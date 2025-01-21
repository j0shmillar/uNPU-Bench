################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../source/image/image_decode_raw.c \
../source/image/image_load.c 

C_DEPS += \
./source/image/image_decode_raw.d \
./source/image/image_load.d 

OBJS += \
./source/image/image_decode_raw.o \
./source/image/image_load.o 


# Each subdirectory must supply rules for building sources it contributes
source/image/%.o: ../source/image/%.c source/image/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: MCU C Compiler'
	arm-none-eabi-gcc -std=gnu99 -D__NEWLIB__ -DPRINTF_FLOAT_ENABLE=1 -DCPU_MCXN947VDF -DCPU_MCXN947VDF_cm33 -DCPU_MCXN947VDF_cm33_core0 -DSDK_DEBUGCONSOLE_UART -DARM_MATH_CM33 -D__FPU_PRESENT=1 -DTF_LITE_STATIC_MEMORY -DMCUXPRESSO_SDK -DSDK_DEBUGCONSOLE=1 -DCR_INTEGER_PRINTF -D__MCUXPRESSO -D__USE_CMSIS -DDEBUG -DLCD_IMPL_FLEXIO=1 -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/source" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/utilities" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/eiq/tensorflow-lite" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/eiq/tensorflow-lite/third_party/flatbuffers/include" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/eiq/tensorflow-lite/third_party/gemmlowp" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/component/lists" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/component/uart" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/drivers" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/device" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/startup" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/eiq/tensorflow-lite/tensorflow/lite/micro/kernels/neutron" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/eiq/tensorflow-lite/third_party/ruy" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/CMSIS" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/eiq/tensorflow-lite/third_party/neutron/common/include" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/eiq/tensorflow-lite/third_party/neutron/driver/include" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/source/image" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/source/model" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/board" -O3 -fno-common -g3 -gdwarf-4 -Wall -Wno-strict-aliasing -mcpu=cortex-m33 -c -ffunction-sections -fdata-sections -fmacro-prefix-map="$(<D)/"= -mcpu=cortex-m33 -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb -D__NEWLIB__ -fstack-usage -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.o)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


clean: clean-source-2f-image

clean-source-2f-image:
	-$(RM) ./source/image/image_decode_raw.d ./source/image/image_decode_raw.o ./source/image/image_load.d ./source/image/image_load.o

.PHONY: clean-source-2f-image

