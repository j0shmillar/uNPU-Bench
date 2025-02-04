################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../source/model/get_top_n.cpp \
../source/model/model.cpp \
../source/model/model_ops_npu.cpp \
../source/model/output_postproc.cpp 

S_SRCS += \
../source/model/model_data.s 

CPP_DEPS += \
./source/model/get_top_n.d \
./source/model/model.d \
./source/model/model_ops_npu.d \
./source/model/output_postproc.d 

OBJS += \
./source/model/get_top_n.o \
./source/model/model.o \
./source/model/model_data.o \
./source/model/model_ops_npu.o \
./source/model/output_postproc.o 


# Each subdirectory must supply rules for building sources it contributes
source/model/%.o: ../source/model/%.cpp source/model/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: MCU C++ Compiler'
	arm-none-eabi-c++ -std=gnu++11 -DCPU_MCXN947VDF -DCPU_MCXN947VDF_cm33 -DCPU_MCXN947VDF_cm33_core0 -DSDK_DEBUGCONSOLE_UART -DARM_MATH_CM33 -D__FPU_PRESENT=1 -DTF_LITE_STATIC_MEMORY -DMCUXPRESSO_SDK -DSDK_DEBUGCONSOLE=1 -D__MCUXPRESSO -D__USE_CMSIS -DDEBUG -D__NEWLIB__ -I"/Users/joshmillar/Desktop/phd/mcu-nn-eval/npu/mcxn947/unet_q/source" -I"/Users/joshmillar/Desktop/phd/mcu-nn-eval/npu/mcxn947/unet_q/utilities" -I"/Users/joshmillar/Desktop/phd/mcu-nn-eval/npu/mcxn947/unet_q/eiq/tensorflow-lite" -I"/Users/joshmillar/Desktop/phd/mcu-nn-eval/npu/mcxn947/unet_q/eiq/tensorflow-lite/third_party/flatbuffers/include" -I"/Users/joshmillar/Desktop/phd/mcu-nn-eval/npu/mcxn947/unet_q/eiq/tensorflow-lite/third_party/gemmlowp" -I"/Users/joshmillar/Desktop/phd/mcu-nn-eval/npu/mcxn947/unet_q/component/lists" -I"/Users/joshmillar/Desktop/phd/mcu-nn-eval/npu/mcxn947/unet_q/component/uart" -I"/Users/joshmillar/Desktop/phd/mcu-nn-eval/npu/mcxn947/unet_q/drivers" -I"/Users/joshmillar/Desktop/phd/mcu-nn-eval/npu/mcxn947/unet_q/device" -I"/Users/joshmillar/Desktop/phd/mcu-nn-eval/npu/mcxn947/unet_q/startup" -I"/Users/joshmillar/Desktop/phd/mcu-nn-eval/npu/mcxn947/unet_q/eiq/tensorflow-lite/tensorflow/lite/micro/kernels/neutron" -I"/Users/joshmillar/Desktop/phd/mcu-nn-eval/npu/mcxn947/unet_q/eiq/tensorflow-lite/third_party/ruy" -I"/Users/joshmillar/Desktop/phd/mcu-nn-eval/npu/mcxn947/unet_q/CMSIS" -I"/Users/joshmillar/Desktop/phd/mcu-nn-eval/npu/mcxn947/unet_q/eiq/tensorflow-lite/third_party/neutron/common/include" -I"/Users/joshmillar/Desktop/phd/mcu-nn-eval/npu/mcxn947/unet_q/eiq/tensorflow-lite/third_party/neutron/driver/include" -I"/Users/joshmillar/Desktop/phd/mcu-nn-eval/npu/mcxn947/unet_q/source/image" -I"/Users/joshmillar/Desktop/phd/mcu-nn-eval/npu/mcxn947/unet_q/source/model" -I"/Users/joshmillar/Desktop/phd/mcu-nn-eval/npu/mcxn947/unet_q/board" -O3 -fno-common -g3 -gdwarf-4 -Wall -fno-rtti -fno-exceptions -Wno-sign-compare -Wno-strict-aliasing -Wno-deprecated-declarations -mcpu=cortex-m33 -c -ffunction-sections -fdata-sections -fmacro-prefix-map="$(<D)/"= -mcpu=cortex-m33 -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb -D__NEWLIB__ -fstack-usage -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.o)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

source/model/%.o: ../source/model/%.s source/model/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: MCU Assembler'
	arm-none-eabi-gcc -c -x assembler-with-cpp -D__NEWLIB__ -I"/Users/joshmillar/Desktop/phd/mcu-nn-eval/npu/mcxn947/unet_q/source" -g3 -gdwarf-4 -mcpu=cortex-m33 -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb -D__NEWLIB__ -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


clean: clean-source-2f-model

clean-source-2f-model:
	-$(RM) ./source/model/get_top_n.d ./source/model/get_top_n.o ./source/model/model.d ./source/model/model.o ./source/model/model_data.o ./source/model/model_ops_npu.d ./source/model/model_ops_npu.o ./source/model/output_postproc.d ./source/model/output_postproc.o

.PHONY: clean-source-2f-model

