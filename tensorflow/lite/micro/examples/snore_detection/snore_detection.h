/*
 * Encapsulate mfcc feature extraction and tflite inference into one package
 * avoid to expose the details of mfcc feature extraction and tflite inference
 * Copyright Li Tongzhi @2023
*/

#ifndef SNORE_DETECTION_H_
#define SNORE_DETECTION_H_
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//initial mfcc feature module
//default parameters are used, 16bit pcm, 8kHz, 32ms frame size, 31ms frame shift, 32 mel filter banks
//if success, return 0, else return -1
int InitializeMFCCFeatures();

//extract mfcc feature from pcm_data
//if success, return 0, else return -1
int ExtractMFCCFeatures(const int16_t* pcm_data, int pcm_data_len,  uint16_t* output_data, size_t* num_samples_read);


/* //initialize tflite model, if success return the handle of  model, else return NULL
 *parameters: 
    @weights_buffer: pointer to tflite model weights
    @model_buffer: pointer to tflite model
    @model_buffer_len: length of tflite model
*/   
void* CreateTFModel(
    void* weights_buffer,
    void* model_buffer, 
    size_t model_buffer_len
  );

/*inference if success, return 0, else return -1
 *parameters: 
    @pModel: pointer to tflite model
    @input_data: pointer to input data
    @input_data_len: length of input data
    @output_data: pointer to output data
    @output_data_len: length of output data
    @model_buffer: pointer to  temporal buffer
    @model_buffer_len: length of temporal buffer
*/
int InferenceTFModel(
                     void* pModel, 
                     const uint16_t* input_data, 
                     int input_data_len, 
                     uint16_t* output_data, 
                     int* output_data_len
                     );
#ifdef __cplusplus
}
#endif 

#endif // SNORE_DETECTION_H_