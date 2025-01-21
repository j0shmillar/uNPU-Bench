################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../utilities/fsl_assert.c \
../utilities/fsl_debug_console.c \
../utilities/fsl_str.c 

S_UPPER_SRCS += \
../utilities/fsl_memcpy.S 

C_DEPS += \
./utilities/fsl_assert.d \
./utilities/fsl_debug_console.d \
./utilities/fsl_str.d 

OBJS += \
./utilities/fsl_assert.o \
./utilities/fsl_debug_console.o \
./utilities/fsl_memcpy.o \
./utilities/fsl_str.o 


# Each subdirectory must supply rules for building sources it contributes
utilities/%.o: ../utilities/%.c utilities/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: MCU C Compiler'
	arm-none-eabi-gcc -std=gnu99 -D__NEWLIB__ -DPRINTF_FLOAT_ENABLE=1 -DCPU_MCXN947VDF -DCPU_MCXN947VDF_cm33 -DCPU_MCXN947VDF_cm33_core0 -DSDK_DEBUGCONSOLE_UART -DARM_MATH_CM33 -D__FPU_PRESENT=1 -DTF_LITE_STATIC_MEMORY -DMCUXPRESSO_SDK -DSDK_DEBUGCONSOLE=1 -DCR_INTEGER_PRINTF -D__MCUXPRESSO -D__USE_CMSIS -DDEBUG -DLCD_IMPL_FLEXIO=1 -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/source" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/utilities" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/eiq/tensorflow-lite" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/eiq/tensorflow-lite/third_party/flatbuffers/include" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/eiq/tensorflow-lite/third_party/gemmlowp" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/component/lists" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/component/uart" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/drivers" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/device" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/startup" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/eiq/tensorflow-lite/tensorflow/lite/micro/kernels/neutron" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/eiq/tensorflow-lite/third_party/ruy" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/CMSIS" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/eiq/tensorflow-lite/third_party/neutron/common/include" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/eiq/tensorflow-lite/third_party/neutron/driver/include" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/source/image" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/source/model" -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/board" -O3 -fno-common -g3 -gdwarf-4 -Wall -Wno-strict-aliasing -mcpu=cortex-m33 -c -ffunction-sections -fdata-sections -fmacro-prefix-map="$(<D)/"= -mcpu=cortex-m33 -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb -D__NEWLIB__ -fstack-usage -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.o)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

utilities/%.o: ../utilities/%.S utilities/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: MCU Assembler'
	arm-none-eabi-gcc -c -x assembler-with-cpp -D__NEWLIB__ -I"/Users/joshmillar/Desktop/phd/npus/mcxn947/dm-fashion-mnist-recognition-on-mcxn947/source" -g3 -gdwarf-4 -mcpu=cortex-m33 -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb -D__NEWLIB__ -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


clean: clean-utilities

clean-utilities:
	-$(RM) ./utilities/fsl_assert.d ./utilities/fsl_assert.o ./utilities/fsl_debug_console.d ./utilities/fsl_debug_console.o ./utilities/fsl_memcpy.o ./utilities/fsl_str.d ./utilities/fsl_str.o

.PHONY: clean-utilities

