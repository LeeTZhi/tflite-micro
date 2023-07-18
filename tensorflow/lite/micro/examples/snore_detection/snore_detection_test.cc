/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <math.h>

#define TEST_SNORE_ONLY 1

#ifndef TEST_SNORE_ONLY
#include "tensorflow/lite/core/c/common.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "tensorflow/lite/micro/examples/snore_detection/mfcc_feature.h"
#else 
#define MicroPrintf printf
#define LOG_INFO(info) MicroPrintf("%s LINE: %d %s\n", __FILE__, __LINE__,  info)
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#endif //TEST_SNORE_ONLY

#include "tensorflow/lite/micro/examples/snore_detection/golden_data.h"
#include "tensorflow/lite/micro/examples/snore_detection/models/snore_detection_float_model_data.h"
#include "tensorflow/lite/micro/examples/snore_detection/models/snore_detection_int8_model_data.h"
#include "tensorflow/lite/micro/examples/snore_detection/snore_detection.h"

#ifndef TEST_SNORE_ONLY
namespace {
using SnoreDetectionOpResolver = tflite::MicroMutableOpResolver<8>;

TfLiteStatus RegisterOps(SnoreDetectionOpResolver& op_resolver) {
  
  TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
  TF_LITE_ENSURE_STATUS(op_resolver.AddAveragePool2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddRelu());
  return kTfLiteOk;
}
}  // namespace

#define LOG_INFO(info) MicroPrintf("%s LINE: %d %s\n", __FILE__, __LINE__,  info)

TfLiteStatus ProfileMemoryAndLatency() {
  tflite::MicroProfiler profiler;
  SnoreDetectionOpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  // Arena size just a round number. The exact arena usage can be determined
  // using the RecordingMicroInterpreter.
  constexpr int kTensorArenaSize = 300*1024;
  uint8_t tensor_arena[kTensorArenaSize];
  constexpr int kNumResourceVariables = 24;

  tflite::RecordingMicroAllocator* allocator(
      tflite::RecordingMicroAllocator::Create(tensor_arena, kTensorArenaSize));
  tflite::RecordingMicroInterpreter interpreter(
      tflite::GetModel(snore_detection_float_tflite), op_resolver, allocator,
      tflite::MicroResourceVariables::Create(allocator, kNumResourceVariables),
      &profiler);

  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());
  TFLITE_CHECK_EQ(interpreter.inputs_size(), 1);
  interpreter.input(0)->data.f[0] = 1.f;
  TF_LITE_ENSURE_STATUS(interpreter.Invoke());

  MicroPrintf("");  // Print an empty new line
  profiler.LogTicksPerTagCsv();

  MicroPrintf("");  // Print an empty new line
  interpreter.GetMicroAllocator().PrintAllocations();
  return kTfLiteOk;
}

TfLiteStatus LoadFloatModelAndPerformInference() {
  const tflite::Model* model =
      ::tflite::GetModel(snore_detection_float_tflite);
  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

  SnoreDetectionOpResolver op_resolver;
  

  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  // Arena size just a round number. The exact arena usage can be determined
  // using the RecordingMicroInterpreter.
  constexpr int kTensorArenaSize = 512*1024;
  uint8_t tensor_arena[kTensorArenaSize];

  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize);
  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

  // Check if the predicted output is within a small range of the
  // expected output
  constexpr int kNumTestValues = 32*32;

  for (int i = 0; i < kNumTestValues; ++i) {
    interpreter.input(0)->data.f[i] = golden_inputs[i];
  }

  TF_LITE_ENSURE_STATUS(interpreter.Invoke());
  float y_pred0 = interpreter.output(0)->data.f[0];
  float y_pred1 = interpreter.output(0)->data.f[1];
  MicroPrintf("y_pred0: %d y_pred1: %d\n", (int32_t)(y_pred0*1000), (int32_t)(y_pred1*1000));
  return kTfLiteOk;
}

TfLiteStatus LoadQuantModelAndPerformInference() {
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model =
      ::tflite::GetModel(snore_detection_int8_tflite);
  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

  SnoreDetectionOpResolver op_resolver;
  
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));
  TF_LITE_ENSURE_STATUS(op_resolver.AddQuantize());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDequantize());

  // Arena size just a round number. The exact arena usage can be determined
  // using the RecordingMicroInterpreter.
  constexpr int kTensorArenaSize = 512*1024;
  uint8_t tensor_arena[kTensorArenaSize];

  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize);

  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

  TfLiteTensor* input = interpreter.input(0);
  TFLITE_CHECK_NE(input, nullptr);

  TfLiteTensor* output = interpreter.output(0);
  TFLITE_CHECK_NE(output, nullptr);

  //float output_scale = output->params.scale;
  //int output_zero_point = output->params.zero_point;

  // Check if the predicted output is within a small range of the
  // expected output

  constexpr int kNumTestValues = 32*32;

  // The int8 values are calculated using the following formula
  // (golden_inputs_float[i] / input->params.scale + input->params.scale)

  //extract feature from pcm_data and then predict
  uint16_t output_data[32*32];
  int audio_size = pcm_data_len;
  int16_t * audio_data = pcm_data;
  int frame_out_dim = 32;
  size_t num_samples_read = 0;
  int i = 0;
  LOG_INFO("start to extract feature");
  TF_LITE_ENSURE_STATUS(InitializeMicroFeatures());
  
  while (audio_size >0 ) {
        GenerateMicroFeatures(audio_data, audio_size, frame_out_dim, output_data+i*frame_out_dim, &num_samples_read);
        audio_data += num_samples_read;
        audio_size -= num_samples_read;
        i++;
  }
  LOG_INFO("extract feature done");
  for ( i = 0; i < kNumTestValues; ++i) {
    input->data.f[i] = static_cast<float>(output_data[i]);
  }
  TF_LITE_ENSURE_STATUS(interpreter.Invoke());
  float y_pred0 = interpreter.output(0)->data.f[0];
  float y_pred1 = interpreter.output(0)->data.f[1];
  MicroPrintf("y_pred0: %d y_pred1: %d\n", (int32_t)(y_pred0*1000), (int32_t)(y_pred1*1000));
  return kTfLiteOk;
}

#endif  // TESTING

int TestSnoreLibFunction() {
  

  void* model = CreateTFModel(snore_detection_float_tflite);
  
  assert(model != NULL);
  //TFLITE_CHECK_NE(model, NULL);

  //Initial MFCC feature
  int ret = InitializeMFCCFeatures();
  
  //extract features
  constexpr int kNumTestValues = 32*32;

  // The int8 values are calculated using the following formula
  // (golden_inputs_float[i] / input->params.scale + input->params.scale)

  //extract feature from pcm_data and then predict
  uint16_t mfcc_output_data[32*32];

  int audio_size = pcm_data_len;
  int16_t * audio_data = pcm_data;
  //int frame_out_dim = 32;
  size_t num_samples_read = 0;
  LOG_INFO("start to extract feature");
  ret = ExtractMFCCFeatures(audio_data, audio_size, mfcc_output_data, &num_samples_read);
  //assert mfcc_out equal golden data
  for ( int i = 0; i < kNumTestValues; ++i) {
    assert(mfcc_output_data[i] == golden_inputs[i]);
  }
  LOG_INFO("extract feature done");
  assert(ret == 0);


  // using the RecordingMicroInterpreter.
  constexpr int kTensorArenaSize = 512*1024;
  uint8_t tensor_arena[kTensorArenaSize];

  uint16_t output[2];
  int output_len = 2;
  ret = InferenceTFModel(model, 
             mfcc_output_data, kNumTestValues,
              output, &output_len,
              tensor_arena, kTensorArenaSize);

  assert(ret == 0);

  MicroPrintf("y_pred0: %d y_pred1: %d\n", (int32_t)(output[0]), (int32_t)(output[1]));
  return 0;
}

int main(int argc, char* argv[]) {
  LOG_INFO("Starting snore detection test...InitializerTarget\n");

#ifndef TEST_SNORE_ONLY
  tflite::InitializeTarget();
  LOG_INFO("Starting snore detection test...\n");
  TF_LITE_ENSURE_STATUS(ProfileMemoryAndLatency());
  LOG_INFO("Memory and latency profiling passed!\n");
  TF_LITE_ENSURE_STATUS(LoadFloatModelAndPerformInference());
  LOG_INFO("Float model inference passed!\n");
  TF_LITE_ENSURE_STATUS(LoadQuantModelAndPerformInference());
  LOG_INFO("Quant model inference passed!\n");
#endif

  assert(TestSnoreLibFunction()==0);
  LOG_INFO("Test snore lib function passed!\n");

  LOG_INFO("~~~ALL TESTS PASSED~~~\n");
  return 0;
}
