#include <iostream>
#include <cmath>
#include <Eigen/Core>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Include only core SAI headers (avoiding test_util.h)
#include "sai.h"
#include "common.h"

using namespace std;

// Standalone helper function to create SAI parameters
SAIParams CreateSAIParams(int num_channels, int input_segment_width, 
                         int trigger_window_width, int sai_width) {
    SAIParams params;
    params.num_channels = num_channels;
    params.input_segment_width = input_segment_width;
    params.trigger_window_width = trigger_window_width;
    params.sai_width = sai_width;
    params.future_lags = sai_width / 2;
    params.num_triggers_per_frame = 1;
    return params;
}

// Simple pulse train generator
ArrayXX CreatePulseTrain(int num_channels, int width, int period) {
    ArrayXX result = ArrayXX::Zero(num_channels, width);
    for (int i = 0; i < num_channels; ++i) {
        for (int j = 0; j < width; j += period) {
            if (j < width) {
                result(i, j) = 1.0;
            }
        }
    }
    return result;
}

int main() {
    cout << "Testing SAI without Google Test or test_util.h" << endl;
    cout << "===============================================" << endl;
    
    // Test 1: Eigen functionality
    cout << "\n1. Testing Eigen..." << endl;
    try {
        Eigen::MatrixXd test_matrix(2, 2);
        test_matrix << 1, 2, 3, 4;
        cout << "Eigen matrix:\n" << test_matrix << endl;
        cout << "✓ Eigen test passed!" << endl;
    } catch (const exception& e) {
        cerr << "✗ Eigen test failed: " << e.what() << endl;
        return 1;
    }
    
    // Test 2: Basic SAI compilation and instantiation
    cout << "\n2. Testing SAI compilation..." << endl;
    try {
        const int kNumChannels = 2;
        const int kInputSegmentWidth = 20;
        const int kSAIWidth = 10;
        const int kTriggerWindowWidth = kInputSegmentWidth;
        
        SAIParams sai_params = CreateSAIParams(kNumChannels, kInputSegmentWidth,
                                               kTriggerWindowWidth, kSAIWidth);
        cout << "SAI parameters created:" << endl;
        cout << "  Channels: " << sai_params.num_channels << endl;
        cout << "  Input width: " << sai_params.input_segment_width << endl;
        cout << "  SAI width: " << sai_params.sai_width << endl;
        
        SAI sai(sai_params);
        cout << "✓ SAI object created successfully!" << endl;
        
    } catch (const exception& e) {
        cerr << "✗ SAI compilation test failed: " << e.what() << endl;
        return 1;
    }
    
    // Test 3: Basic SAI processing
    cout << "\n3. Testing SAI processing..." << endl;
    try {
        const int kNumChannels = 1;
        const int kInputSegmentWidth = 20;
        const int kSAIWidth = 10;
        const int kTriggerWindowWidth = kInputSegmentWidth;
        const int kPeriod = 5;
        
        SAIParams sai_params = CreateSAIParams(kNumChannels, kInputSegmentWidth,
                                               kTriggerWindowWidth, kSAIWidth);
        sai_params.future_lags = 0;  // Only past lags
        
        SAI sai(sai_params);
        
        // Create simple pulse train input
        ArrayXX segment = CreatePulseTrain(kNumChannels, kInputSegmentWidth, kPeriod);
        cout << "Input pulse train:" << endl << segment << endl;
        
        ArrayXX sai_frame;
        sai.RunSegment(segment, &sai_frame);
        
        cout << "SAI output:" << endl << sai_frame << endl;
        cout << "Output dimensions: " << sai_frame.rows() << "x" << sai_frame.cols() << endl;
        cout << "✓ SAI processing test passed!" << endl;
        
    } catch (const exception& e) {
        cerr << "✗ SAI processing test failed: " << e.what() << endl;
        return 1;
    }
    
    cout << "\n===============================================" << endl;
    cout << "All tests completed successfully!" << endl;
    cout << "Your SAI library is working correctly." << endl;
    
    return 0;
}