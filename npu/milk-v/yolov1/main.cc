#include <iostream>
#include <thread>
#include <vector>
#include <cstdio>
#include "utils.h"
#include "model.h" 

using std::cerr;
using std::cout;
using std::endl;

std::atomic<bool> isp_stop(false);

#define GRID_SIZE 12
#define NUM_CLASSES 2  // Based on the output tensor shape shown
#define NUM_CONFIDENCE 10 // Based on the slice operations shown
#define OUTPUT_STRIDE (NUM_CLASSES + NUM_CONFIDENCE)

#define CONV_OUT_H 12
#define CONV_OUT_W 12
#define CONV_OUT_C 12

#define OUT_H 12
#define OUT_W 12
#define SIGMOID_CH 1 
#define SOFTMAX_CH 1 
#define TOTAL_CH (SIGMOID_CH + SOFTMAX_CH)
#define MAX_DETECTIONS (OUT_H * OUT_W)


typedef struct {
    float x, y;    
    float confidence;    
    float class_probs[NUM_CLASSES];  
    int class_id;     
} detection_t;

void softmax(const float* input, float* output, int length) {
    float max_val = input[0];
    for (int i = 1; i < length; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    float inv_sum = 1.0f / sum;
    for (int i = 0; i < length; i++) {
        output[i] *= inv_sum;
    }
}

float sigmoid(float x) {
    if (x < -5.0f) return 0.0f;
    if (x > 5.0f) return 1.0f;
    return 1.0f / (1.0f + expf(-x));
}


void process_yolo_output(const std::vector<float*>& output_array, float* final_output, int grid_size) {
    // Each grid cell has:
    // - 2 confidence scores from first slice
    // - 10 class probabilities from second slice
    const int confidence_stride = grid_size * grid_size * 2;  // Size of confidence tensor
    
    // Process each grid cell
    for (int y = 0; y < grid_size; y++) {
        for (int x = 0; x < grid_size; x++) {
            printf("here3\n");
            int grid_idx = y * grid_size + x;
            int out_idx = grid_idx * OUTPUT_STRIDE;
            
            // Get confidence scores (first 2 values per grid cell)
            for (int i = 0; i < NUM_CONFIDENCE; i++) {
                float conf = output_array[0][grid_idx * 2 + i];  // From first part of output
                final_output[out_idx + i] = sigmoid(conf);
            }
            
            // Get class probabilities (next 10 values per grid cell)
            float class_inputs[NUM_CLASSES];
            for (int i = 0; i < NUM_CLASSES; i++) {
                class_inputs[i] = output_array[1][confidence_stride + grid_idx * NUM_CLASSES + i];
            }
            
            // Apply softmax to class probabilities
            softmax(class_inputs, &final_output[out_idx + NUM_CONFIDENCE], NUM_CLASSES);
        }
    }
}


void extract_detections(const float* processed_output, 
                       detection_t* detections,
                       int* num_detections,
                       float confidence_threshold,
                       int grid_size) {
    *num_detections = 0;
    
    for (int y = 0; y < grid_size; y++) {
        for (int x = 0; x < grid_size; x++) {
            int grid_idx = y * grid_size + x;
            int out_idx = grid_idx * OUTPUT_STRIDE;
            
            float conf1 = processed_output[out_idx];
            float conf2 = processed_output[out_idx + 1];
            
            if (conf1 > confidence_threshold) {
                detection_t* det = &detections[*num_detections];
                
                det->x = ((float)x + 0.5f) / grid_size;
                det->y = ((float)y + 0.5f) / grid_size;
                det->confidence = conf1;
                
                float max_prob = 0.0f;
                for (int i = 0; i < NUM_CLASSES; i++) {
                    float prob = processed_output[out_idx + NUM_CONFIDENCE + i];
                    det->class_probs[i] = prob;  
                    if (prob > max_prob) {
                        max_prob = prob;
                        det->class_id = i;
                    }
                }
                
                (*num_detections)++;
            }
            
            if (conf2 > confidence_threshold) {
                detection_t* det = &detections[*num_detections];
                
                det->x = ((float)x + 0.5f) / grid_size;
                det->y = ((float)y + 0.5f) / grid_size;
                det->confidence = conf2;
                
                float max_prob = 0.0f;
                for (int i = 0; i < NUM_CLASSES; i++) {
                    float prob = processed_output[out_idx + NUM_CONFIDENCE + i];
                    det->class_probs[i] = prob;  
                    if (prob > max_prob) {
                        max_prob = prob;
                        det->class_id = i;
                    }
                }
                
                (*num_detections)++;
            }
        }
    }
}

void process_random_image(char *argv[]) 
{    
    const int img_width = 96;
    const int img_height = 96;
    
    // Generate random image (flat 1D array of unsigned chars)
    std::vector<unsigned char> random_img(img_width * img_height * 3);
    for (auto& pixel : random_img) {
        pixel = rand() % 256;  // Random value between 0 and 255
    }

    Model model(argv[1]); 
    
    // Instead of pre_process, directly use random image
    model.inference(random_img.data(), img_width, img_height);  // Passing data directly for inference

    auto results = model.get_output();
    const auto& output_shapes = model.get_output_shapes(); 

    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << "Output tensor " << i << ": ";
        for (int j = 0; j < 10 && j < output_shapes[i][0]; ++j) {
            std::cout << results[i][j] << " ";
        }
        std::cout << "..." << std::endl;
    }

    float processed_output[GRID_SIZE * GRID_SIZE * OUTPUT_STRIDE];
    detection_t detections[GRID_SIZE * GRID_SIZE * 2];
    int num_detections;

    process_yolo_output(results, processed_output, GRID_SIZE);

    float confidence_threshold = 0.5f;
    extract_detections(processed_output, detections, &num_detections, confidence_threshold, GRID_SIZE);

    std::printf("\nDetections found: %d\n", num_detections);
    for (int i = 0; i < num_detections; i++) {
        std::printf("Detection %d: (x=%.3f, y=%.3f) conf=%.3f class=%d score=%.3f\n",
            i,
            detections[i].x,
            detections[i].y,
            detections[i].confidence,
            detections[i].class_id,
            detections[i].class_probs[detections[i].class_id]);
    }
}


int main(int argc, char *argv[])
{
    std::cout << "case " << argv[0] << " built at " << __DATE__ << " " << __TIME__ << std::endl;
    process_random_image(argv);
    return 0;
}
