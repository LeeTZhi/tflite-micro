#include <stdio.h>
#include "tensorflow/lite/micro/examples/snore_detection/mfcc_feature.h"

int main(int argc, char* argv[]) {
    tflite::InitializeTarget();

    size_t num_samples_read = 0;

    const int output_size = 32*32;
    uint16_t output[output_size];
    int i = 0;
    int frame_len = 32*8;
    int frame_shift = 31*8;
    int frame_out = 32;
    int frame_count = (pcm_data_len-frame_len) / frame_shift + 1;
    printf("pcm_data_len: %d frame_count: %d\n", pcm_data_len, frame_count);
    InitializeMicroFeatures();
    
    int audio_size = static_cast<int>(pcm_data_len);
    int16_t* audio_data = pcm_data;

    while (audio_size >0 ) {
        GenerateMicroFeatures(audio_data, audio_size, frame_out, output+i*frame_out, &num_samples_read);
        audio_data += num_samples_read;
        audio_size -= num_samples_read;
        i++;
    }
    
    for ( i = 0; i < output_size; i++) {
        if ( i % 32 == 0 ) {
            printf("idx: %d ", i/32);
        }
        printf("%d ", output[i]);
        if ( i % 32 == 31) {
            printf("\n");
        }
    }
    return 0;
}