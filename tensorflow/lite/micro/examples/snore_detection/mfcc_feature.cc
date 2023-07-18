


#include <cmath>
#include <cstring>

#include "tensorflow/lite/experimental/microfrontend/lib/frontend.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend_util.h"
#include "tensorflow/lite/micro/examples/micro_speech/micro_features/micro_model_settings.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/examples/snore_detection/mfcc_feature.h"



namespace {
    FrontendState g_micro_features_state;
    bool g_is_first_time = true;
}


#define LOG_INFO(info) MicroPrintf("%s LINE: %d %s\n", __FILE__, __LINE__,  info)

TfLiteStatus InitializeMicroFeatures() {
  FrontendConfig config;
  
  config.window.size_ms = 32;
  config.window.step_size_ms = 31;
  config.noise_reduction.smoothing_bits = 10;
  config.filterbank.num_channels = 32;
  config.filterbank.lower_band_limit = 20.0;
  config.filterbank.upper_band_limit = 3800.0;
  config.noise_reduction.smoothing_bits = 10;
  config.noise_reduction.even_smoothing = 0.025;
  config.noise_reduction.odd_smoothing = 0.06;
  config.noise_reduction.min_signal_remaining = 0.05;
  config.pcan_gain_control.enable_pcan = 1;
  config.pcan_gain_control.strength = 0.95;
  config.pcan_gain_control.offset = 80.0;
  config.pcan_gain_control.gain_bits = 21;
  config.log_scale.enable_log = 1;
  config.log_scale.scale_shift = 6;
  if (!FrontendPopulateState(&config, &g_micro_features_state,
                             8000)) {
    MicroPrintf("FrontendPopulateState() failed");
    return kTfLiteError;
  }
  
  g_is_first_time = true;
  return kTfLiteOk;
}

// This is not exposed in any header, and is only used for testing, to ensure
// that the state is correctly set up before generating results.
void SetMicroFeaturesNoiseEstimates(const uint32_t* estimate_presets) {
  for (int i = 0; i < g_micro_features_state.filterbank.num_channels; ++i) {
    g_micro_features_state.noise_reduction.estimate[i] = estimate_presets[i];
  }
}

TfLiteStatus GenerateMicroFeatures(const int16_t* input, int input_size,
                                   int output_size, uint16_t* output,
                                   size_t* num_samples_read) {
  const int16_t* frontend_input;
  if (g_is_first_time) {
    frontend_input = input;
    g_is_first_time = false;
  } else {
    frontend_input = input;
  }

  FrontendOutput frontend_output = FrontendProcessSamples(
      &g_micro_features_state, frontend_input, input_size, num_samples_read);

  for (size_t i = 0; i < frontend_output.size; ++i) {
    output[i] = frontend_output.values[i];
  }

  return kTfLiteOk;
}