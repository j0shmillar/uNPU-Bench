################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../eiq/tensorflow-lite/tensorflow/lite/micro/debug_log.cpp 

CPP_DEPS += \
./eiq/tensorflow-lite/tensorflow/lite/micro/debug_log.d 

OBJS += \
./eiq/tensorflow-lite/tensorflow/lite/micro/debug_log.o 


# Each subdirectory must supply rules for building sources it contributes
eiq/tensorflow-lite/tensorflow/lite/micro/%.o: ../eiq/tensorflow-lite/tensorflow/lite/micro/%.cpp eiq/tensorflow-lite/tensorflow/lite/micro/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: MCU C++ Compiler'
	arm-none-eabi-c++ -std=gnu++11 -DCPU_MCXN947VDF -DPRINTF_FLOAT_ENABLE -DCPU_MCXN947VDF_cm33 -DCPU_MCXN947VDF_cm33_core0 -DSDK_DEBUGCONSOLE_UART -DARM_MATH_CM33 -D__FPU_PRESENT=1 -DTF_LITE_STATIC_MEMORY -DMCUXPRESSO_SDK -DSDK_DEBUGCONSOLE=1 -D__MCUXPRESSO -D__USE_CMSIS -DDEBUG -D__NEWLIB__ -I"/Users/joshmillar/Desktop/phd/mcu-nn-eval/src/templates/mcxn947/source" -I"/Users/joshmillar/Desktop/phd/mcu-nn-eval/src/templates/mcxn947/utilities" -I"/Users/joshmillar/Desktop/phd/mcu-nn-eval/src/templates/mcxn947/eiq/tensorflow-lite" -I"/Users/joshmillar/Desktop/phd/mcu-nn-eval/src/templates/mcxn947/eiq/tensorflow-lite/third_party/flatbuffers/include" -I"/Users/joshmillar/Desktop/phd/mcu-nn-eval/src/templates/mcxn947/eiq/tensorflow-lite/third_party/gemmlowp" -I"/Users/joshmillar/Desktop/phd/mcu-nn-eval/src/templates/mcxn947/component/lists" -I"/Users/joshmillar/Desktop/phd/mcu-nn-eval/src/templates/mcxn947/component/uart" -I"/Users/joshmillar/Desktop/phd/mcu-nn-eval/src/templates/mcxn947/drivers" -I"/Users/joshmillar/Desktop/phd/mcu-nn-eval/src/templates/mcxn947/device" -I"/Users/joshmillar/Desktop/phd/mcu-nn-eval/src/templates/mcxn947/startup" -I"/Users/joshmillar/Desktop/phd/mcu-nn-eval/src/templates/mcxn947/eiq/tensorflow-lite/tensorflow/lite/micro/kernels/neutron" -I"/Users/joshmillar/Desktop/phd/mcu-nn-eval/src/templates/mcxn947/eiq/tensorflow-lite/third_party/ruy" -I"/Users/joshmillar/Desktop/phd/mcu-nn-eval/src/templates/mcxn947/CMSIS" -I"/Users/joshmillar/Desktop/phd/mcu-nn-eval/src/templates/mcxn947/eiq/tensorflow-lite/third_party/neutron/common/include" -I"/Users/joshmillar/Desktop/phd/mcu-nn-eval/src/templates/mcxn947/eiq/tensorflow-lite/third_party/neutron/driver/include" -I"/Users/joshmillar/Desktop/phd/mcu-nn-eval/src/templates/mcxn947/source/model" -I"/Users/joshmillar/Desktop/phd/mcu-nn-eval/src/templates/mcxn947/board" -O3 -fno-common -g3 -gdwarf-4 -Wall -fno-rtti -fno-exceptions -Wno-sign-compare -Wno-strict-aliasing -Wno-deprecated-declarations -mcpu=cortex-m33 -c -ffunction-sections -fdata-sections -fmacro-prefix-map="$(<D)/"= -mcpu=cortex-m33 -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb -D__NEWLIB__ -fstack-usage -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.o)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


clean: clean-eiq-2f-tensorflow-2d-lite-2f-tensorflow-2f-lite-2f-micro

clean-eiq-2f-tensorflow-2d-lite-2f-tensorflow-2f-lite-2f-micro:
	-$(RM) ./eiq/tensorflow-lite/tensorflow/lite/micro/debug_log.d ./eiq/tensorflow-lite/tensorflow/lite/micro/debug_log.o

.PHONY: clean-eiq-2f-tensorflow-2d-lite-2f-tensorflow-2f-lite-2f-micro

