SNORE_DETECTION_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/snore_detection/snore_detection_test.cc

SNORE_DETECTION_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/snore_detection/snore_detection_test.cc

SNORE_DETECTION_HDRS :=

SNORE_DETECTION_GENERATOR_INPUTS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/snore_detection/models/snore_detection_float.tflite \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/snore_detection/models/snore_detection_int8.tflite

SNORE_DETECTION_GENERATED_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/snore_detection/models/snore_detection_float_model_data.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/snore_detection/models/snore_detection_int8_model_data.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/snore_detection/mfcc_feature.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/snore_detection/golden_data.cc

SNORE_DETECTION_GENERATED_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/snore_detection/models/snore_detection_float_model_data.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/snore_detection/models/snore_detection_int8_model_data.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/snore_detection/golden_data.h

# Add sources and headers generated from $(SNORE_DETECTION_GENERATOR_INPUTS).
SNORE_DETECTION_SRCS += $(SNORE_DETECTION_GENERATED_SRCS)
SNORE_DETECTION_HDRS += $(SNORE_DETECTION_GENERATED_HDRS)

INCLUDES += \
 -I$(MAKEFILE_DIR)/downloads/kissfft

KISSFFT_LIB_SRCS :=

KISSFFT_LIB_HDRS := \
$(MAKEFILE_DIR)/downloads/kissfft/COPYING \
$(MAKEFILE_DIR)/downloads/kissfft/kiss_fft.c \
$(MAKEFILE_DIR)/downloads/kissfft/kiss_fft.h \
$(MAKEFILE_DIR)/downloads/kissfft/_kiss_fft_guts.h \
$(MAKEFILE_DIR)/downloads/kissfft/tools/kiss_fftr.c \
$(MAKEFILE_DIR)/downloads/kissfft/tools/kiss_fftr.h


MICRO_FEATURES_LIB_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/fft.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/fft_util.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/filterbank.c \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/filterbank_util.c \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/frontend.c \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/frontend_util.c \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/kiss_fft_int16.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/log_lut.c \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/log_scale.c \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/log_scale_util.c \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/noise_reduction.c \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/noise_reduction_util.c \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control.c \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control_util.c \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/window.c \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/window_util.c \
$(KISSFFT_LIB_SRCS)

MICRO_FEATURES_LIB_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/bits.h \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/fft.h \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/fft_util.h \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/filterbank.h \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/filterbank_util.h \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/frontend.h \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/frontend_util.h \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/kiss_fft_common.h \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/kiss_fft_int16.h \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/log_lut.h \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/log_scale.h \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/log_scale_util.h \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/noise_reduction.h \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/noise_reduction_util.h \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control.h \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control_util.h \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/window.h \
$(TENSORFLOW_ROOT)tensorflow/lite/experimental/microfrontend/lib/window_util.h \
$(KISSFFT_LIB_HDRS)

#MFCC FEATURE LIB
MFCC_FEATURE_LIB := libmfcc_feature.a
MFCC_FEATURE_LIB_PATH := $(LIBDIR)$(MFCC_FEATURE_LIB)
MFCC_FEATURE_LIB_OBJS := $(addprefix $(CORE_OBJDIR), \
$(patsubst %.S,%.o,$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(MICRO_FEATURES_LIB_SRCS)))))

$(MFCC_FEATURE_LIB_PATH): $(MFCC_FEATURE_LIB_OBJS) 
	@mkdir -p $(dir $@)
	$(AR) $(ARFLAGS) $(MFCC_FEATURE_LIB_PATH) $(MFCC_FEATURE_LIB_OBJS)

lib_mfcc: $(MFCC_FEATURE_LIB_PATH)

#snore detection lib
SNORE_DETECTION_LIB_SRCS := $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/snore_detection/mfcc_feature.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/snore_detection/snore_detection.cc
SNORE_DETECTION_LIB_OBJS := $(addprefix $(CORE_OBJDIR), \
$(patsubst %.S,%.o,$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(SNORE_DETECTION_LIB_SRCS)))))
SNORE_DETECTION_LIB := libsnore_detection.a
SNORE_DETECTION_LIB_PATH := $(LIBDIR)$(SNORE_DETECTION_LIB)
$(SNORE_DETECTION_LIB_PATH): $(SNORE_DETECTION_LIB_OBJS)
	@mkdir -p $(dir $@)
	$(AR) $(ARFLAGS) $(SNORE_DETECTION_LIB_PATH) $(SNORE_DETECTION_LIB_OBJS)
lib_snore_detection:$(SNORE_DETECTION_LIB_PATH)

#test code
FEATURE_TEST_SRCS :=  $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/snore_detection/golden_data.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/snore_detection/mfcc_feature_test.cc 

FEATURE_TEST_OBJS := $(addprefix $(CORE_OBJDIR), \
$(patsubst %.S,%.o,$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(FEATURE_TEST_SRCS)))))


snore_det_all: $(FEATURE_TEST_OBJS) $(MFCC_FEATURE_LIB_PATH)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $@ \
	$(FEATURE_TEST_OBJS)  $(MFCC_FEATURE_LIB_PATH) $(MICROLITE_LIB_PATH)  $(LDFLAGS) $(MICROLITE_LIBS)

SNORE_DETECTION_TEST_SRCS += $(SNORE_DETECTION_GENERATED_SRCS)

SNORE_DETECTION_SRCS += $(MICRO_FEATURES_LIB_SRCS)

# Tests loading and running the sine model.
$(eval $(call microlite_test,snore_detection_test,\
$(SNORE_DETECTION_TEST_SRCS),,$(SNORE_DETECTION_GENERATOR_INPUTS)))

# Builds a standalone binary.
$(eval $(call microlite_test,snore_detection,\
$(SNORE_DETECTION_SRCS),,$(SNORE_DETECTION_GENERATOR_INPUTS)))


$(eval $(call microlite_test,mfcc_feature_test,\
$(FEATURE_TEST_SRCS),,$(SNORE_DETECTION_GENERATOR_INPUTS)))

SNORE_DETECTION_TEST_OBJS := $(addprefix $(CORE_OBJDIR), \
$(patsubst %.S,%.o,$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(SNORE_DETECTION_TEST_SRCS)))))

snore_detection_test1: $(SNORE_DETECTION_TEST_OBJS) $(MFCC_FEATURE_LIB_PATH) $(SNORE_DETECTION_LIB_PATH)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(BINDIR)$@ \
    $(SNORE_DETECTION_TEST_OBJS) $(SNORE_DETECTION_LIB_PATH) $(MFCC_FEATURE_LIB_PATH)  $(MICROLITE_LIB_PATH)  \
	$(LDFLAGS)  $(MICROLITE_LIBS) 

$(eval $(call microlite_test,mfcc_feature,\
$(FEATURE_TEST_SRCS),,$(SNORE_DETECTION_GENERATOR_INPUTS)))


list_snore_detection_example_sources:
	@echo $(SNORE_DETECTION_SRCS)

list_snore_detection_example_headers:
	@echo $(SNORE_DETECTION_HDRS)
