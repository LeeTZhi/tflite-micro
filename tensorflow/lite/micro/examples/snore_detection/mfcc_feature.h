#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_MFCC_FEATURE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_MFCC_FEATURE_H_

#include "tensorflow/lite/c/common.h"


TfLiteStatus InitializeMicroFeatures();

TfLiteStatus GetMicroFeatureSlice(uint8_t* signal_block, int start_sample,
                                  int duration_ms, int feature_size,
                                  int8_t* feature_data);
TfLiteStatus GenerateMicroFeatures(const int16_t* input, int input_size,
                                   int output_size, uint16_t* output,
                                   size_t* num_samples_read);

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_MFCC_FEATURE_H_                              