/*
 * Implementation of the snore detection.h
 * Encapsulate mfcc feature extraction and tflite inference into one package
*/

#include <cmath>
#include <cstring>

#include "tensorflow/lite/experimental/microfrontend/lib/frontend.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend_util.h"
#include "tensorflow/lite/micro/micro_log.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "tensorflow/lite/micro/examples/snore_detection/snore_detection.h"
#include "tensorflow/lite/micro/examples/snore_detection/mfcc_feature.h"

#define LOG_INFO(info) MicroPrintf("%s LINE: %d %s\n", __FILE__, __LINE__,  info)

#define TF_LITE_ENSURE_NULL(x) \
  do {                          \
      const TfLiteStatus s = (x); \
      if (s != kTfLiteOk) {  \
        MicroPrintf("Error at %s:%d\n", __FILE__, __LINE__); \
        return NULL; \
      } \
  } while (0)

namespace SnoreDetection {
  using SnoreDetectionOpResolver = tflite::MicroMutableOpResolver<10>;

  typedef struct model_struct {
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    void* model_buffer = nullptr;
    size_t model_buffer_len = 0;
  } model_t;
}  // namespace

SnoreDetection::SnoreDetectionOpResolver op_resolver;

extern "C" int InitializeMFCCFeatures() {
  TfLiteStatus mfcc_status = InitializeMicroFeatures();
  if (mfcc_status == kTfLiteOk) {
    return 0;
  } else {
    MicroPrintf("MFCC feature initialization failed: %s %s %d\n", __FILE__, __func__, __LINE__);
    return -1;
  }
}

//Loop extract features
extern "C" int ExtractMFCCFeatures(const int16_t* pcm_data, int pcm_data_len, uint16_t* output_data, size_t* num_samples_read) {
    int audio_size = pcm_data_len;
    int16_t * audio_data = const_cast<int16_t*>(pcm_data);
    int frame_out_dim = 32;
    int i = 0;
    LOG_INFO("start to extract feature");
    size_t num_sample_used = 0;
    while (audio_size >0 ) {
        GenerateMicroFeatures(audio_data, audio_size, frame_out_dim, output_data+i*frame_out_dim, &num_sample_used);
        audio_data += num_sample_used;
        audio_size -= num_sample_used;
        i++;
    }
    *num_samples_read = num_sample_used;
    LOG_INFO("extract feature done");
  return 0;
}

//initialize tflite model
extern "C" void* CreateTFModel( 
  void* weights_buffer, 
  void* model_buffer, 
  size_t model_buffer_len
  ) {
    
    SnoreDetection::model_t* pWrapper =  (SnoreDetection::model_t*)malloc(sizeof(SnoreDetection::model_t));
    if ( !pWrapper ) {
      MicroPrintf("Error at %s:%d\n", __FILE__, __LINE__);
      return NULL;
    }
    pWrapper->model =
      ::tflite::GetModel(weights_buffer);
    TFLITE_CHECK_EQ(pWrapper->model->version(), TFLITE_SCHEMA_VERSION);

    // Build an interpreter to run the model with.
    
    TF_LITE_ENSURE_NULL(op_resolver.AddConv2D());
    TF_LITE_ENSURE_NULL(op_resolver.AddFullyConnected());
    TF_LITE_ENSURE_NULL(op_resolver.AddReshape());
    TF_LITE_ENSURE_NULL(op_resolver.AddSoftmax());
    TF_LITE_ENSURE_NULL(op_resolver.AddAveragePool2D());
    TF_LITE_ENSURE_NULL(op_resolver.AddRelu());
    TF_LITE_ENSURE_NULL(op_resolver.AddDepthwiseConv2D());
    TF_LITE_ENSURE_NULL(op_resolver.AddQuantize());
    TF_LITE_ENSURE_NULL(op_resolver.AddDequantize());

    pWrapper->interpreter = new tflite::MicroInterpreter(pWrapper->model, op_resolver, (uint8_t*)model_buffer, model_buffer_len);

    // Allocate memory from the tensor_arena for the model's tensors.
    TF_LITE_ENSURE_NULL(pWrapper->interpreter->AllocateTensors());
    
    return (void*)pWrapper;
}

//inference
extern "C" int InferenceTFModel(
                     void* pModel, 
                     const uint16_t* input_data, 
                     int input_data_len, 
                     uint16_t* output_data, 
                     int* output_data_len
                     ) {
    SnoreDetection::model_t* pWrapper = (SnoreDetection::model_t*)pModel;
    if ( !pWrapper ) {
      MicroPrintf("Error at %s:%d\n", __FILE__, __LINE__);
      return -1;
    }
    
    // set input
    TfLiteTensor* input = pWrapper->interpreter->input(0);
    for ( int i = 0; i < input_data_len; ++i) {
      input->data.f[i] = static_cast<float>(input_data[i]);
    }

    //invoke inference
    TF_LITE_ENSURE_STATUS(pWrapper->interpreter->Invoke());
#ifdef __linux__
    fprintf(stderr, "usage: %ld\n", pWrapper->interpreter->arena_used_bytes());
#endif
    //get output
    TfLiteTensor* output = pWrapper->interpreter->output(0);
    int output_len = output->bytes / sizeof(uint16_t);
    for (int i = 0; i < output_len; ++i) {
      output_data[i] = static_cast<uint16_t>(output->data.f[i]*1024.0f);
    }
    return 0;
}



