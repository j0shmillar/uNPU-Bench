if(NOT DEFINED FPU)
	set(FPU "-mfpu=fpv5-sp-d16 -mfloat-abi=hard")
endif()

if(${LIBRARY_TYPE} STREQUAL "REDLIB")
	set(SPECS "-specs=redlib.specs")
elseif(${LIBRARY_TYPE} STREQUAL "NEWLIB_NANO")
	set(SPECS "--specs=nano.specs")
endif()

if(NOT DEFINED DEBUG_CONSOLE_CONFIG)
	set(DEBUG_CONSOLE_CONFIG "-DSDK_DEBUGCONSOLE=1")
endif()

set(CMAKE_ASM_FLAGS_DEBUG " \
    ${CMAKE_ASM_FLAGS_DEBUG} \
    ${FPU} \
    -mcpu=cortex-m33 \
    -mthumb \
")

set(CMAKE_C_FLAGS_DEBUG " \
    ${CMAKE_C_FLAGS_DEBUG} \
    ${FPU} \
    ${DEBUG_CONSOLE_CONFIG} \
    -std=gnu99 \
    -DPRINTF_FLOAT_ENABLE \
    -DCPU_MCXN947VDF \
    -DCPU_MCXN947VDF_cm33 \
    -DCPU_MCXN947VDF_cm33_core0 \
    -DSDK_DEBUGCONSOLE_UART \
    -DARM_MATH_CM33 \
    -D__FPU_PRESENT=1 \
    -DTF_LITE_STATIC_MEMORY \
    -DMCUXPRESSO_SDK \
    -DCR_INTEGER_PRINTF \
    -D__MCUXPRESSO \
    -D__USE_CMSIS \
    -DDEBUG \
    -DLCD_IMPL_FLEXIO=1 \
    -O3 \
    -fno-common \
    -fmerge-constants \
    -g3 \
    -gdwarf-4 \
    -Wno-strict-aliasing -mcpu=cortex-m33 -ffunction-sections -fdata-sections \
    -fstack-usage \
    -mcpu=cortex-m33 \
    -mthumb \
")

set(CMAKE_CXX_FLAGS_DEBUG " \
    ${CMAKE_CXX_FLAGS_DEBUG} \
    ${FPU} \
    ${DEBUG_CONSOLE_CONFIG} \
    -std=gnu++11 \
    -DCPU_MCXN947VDF \
    -DPRINTF_FLOAT_ENABLE \
    -DCPU_MCXN947VDF_cm33 \
    -DCPU_MCXN947VDF_cm33_core0 \
    -DSDK_DEBUGCONSOLE_UART \
    -DARM_MATH_CM33 \
    -D__FPU_PRESENT=1 \
    -DTF_LITE_STATIC_MEMORY \
    -DMCUXPRESSO_SDK \
    -D__MCUXPRESSO \
    -D__USE_CMSIS \
    -DDEBUG \
    -O3 \
    -fno-common \
    -fmerge-constants \
    -g3 \
    -gdwarf-4 \
    -Wall \
    -fno-rtti -fno-exceptions -Wno-sign-compare -Wno-strict-aliasing -Wno-deprecated-declarations -mcpu=cortex-m33 -ffunction-sections -fdata-sections \
    -fstack-usage \
    -mcpu=cortex-m33 \
    -mthumb \
")

set(CMAKE_EXE_LINKER_FLAGS_DEBUG " \
    ${CMAKE_EXE_LINKER_FLAGS_DEBUG} \
    ${FPU} \
    ${SPECS} \
    -nostdlib \
    -Xlinker \
    -no-warn-rwx-segments \
    -Xlinker \
    -Map=output.map \
    -Xlinker \
    --gc-sections \
    -Xlinker \
    -print-memory-usage \
    -Xlinker \
    --sort-section=alignment \
    -Xlinker \
    --cref \
    -mcpu=cortex-m33 \
    -mthumb \
    -T\"${ProjDirPath}/yolov1_Debug.ld\" \
")
