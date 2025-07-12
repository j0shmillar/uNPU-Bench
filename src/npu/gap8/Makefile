# Copyright (C) 2017 GreenWaves Technologies
# All rights reserved.

# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE file for details.

ifndef GAP_SDK_HOME
  $(error Source sourceme in gap_sdk first)
endif

MODEL_PREFIX=resnet
TRAINED_MODEL = model/resnet_full_integer_quant.tflite
AT_INPUT_WIDTH=32
AT_INPUT_HEIGHT=32
AT_INPUT_COLORS=3
IMAGES = images

LOAD_QUANTIZATION= #-q #to load a tflite quantized model
IMAGE=$(CURDIR)/images/random_3x32x32.png

io=host

MAIN = $(MODEL_PREFIX).c
MODEL_FP16 ?= 0
ifeq ($(MODEL_FP16), 1)
  NNTOOL_SCRIPT = model/nntool_script_fp16
  MODEL_SUFFIX = _FP16
  MAIN = $(MODEL_PREFIX)_fp16.c
else
QUANT_BITS?=8
BUILD_DIR=BUILD_SQ8BIT
ifeq ($(QUANT_BITS), 8)
	MODEL_SQ8=1
  NNTOOL_SCRIPT=model/nntool_script
  MODEL_SUFFIX = _SQ8BIT
else
  MODEL_POW2=1
  ifeq ($(QUANT_BITS),16)
    NNTOOL_SCRIPT=model/nntool_script16
    MODEL_SUFFIX = _16BIT
  else
    $(error Don\'t know how to build with this bit width)
  endif
endif
endif

$(info Building GAP8 mode with $(QUANT_BITS) bit quantization)

include ../common/model_decl.mk

APP = resnet
APP_SRCS += $(MAIN) $(MODEL_GEN_C) $(MODEL_COMMON_SRCS) $(CNN_LIB)

APP_CFLAGS += -gdwarf-2 -gstrict-dwarf -O3 -mno-memcpy -fno-tree-loop-distribute-patterns
APP_CFLAGS += -I. -I$(MODEL_COMMON_INC) -I$(TILER_EMU_INC) -I$(TILER_INC) $(CNN_LIB_INCLUDE) -I$(MODEL_BUILD)
APP_CFLAGS += -DPERF -DAT_MODEL_PREFIX=$(MODEL_PREFIX) $(MODEL_SIZE_CFLAGS)
APP_CFLAGS += -DSTACK_SIZE=$(CLUSTER_STACK_SIZE) -DSLAVE_STACK_SIZE=$(CLUSTER_SLAVE_STACK_SIZE)
APP_CFLAGS += -AT_IMAGE=$(IMAGE)

READFS_FILES=$(abspath $(MODEL_TENSORS))

# all depends on the model
all:: model

clean:: clean_model

include ../common/model_rules.mk
$(info APP_SRCS... $(APP_SRCS))
$(info APP_CFLAGS... $(APP_CFLAGS))
include $(RULES_DIR)/pmsis_rules.mk
