// real_time_carfac_sai.cpp
// Build prerequisites: Eigen, OpenCV (and your CARFAC C++ headers/libs).
// Example compile (linux, adjust include/lib paths):
// g++ real_time_carfac_sai.cpp -O2 -std=c++17 -I/path/to/eigen -I/path/to/carfac/includes `pkg-config --cflags --libs opencv4` -o realtime_sai

#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

// Include CARFAC headers - adjust paths to your local build
#include "carfac.h"
#include "car.h"
#include "agc.h"
#include "ihc.h"
#include "common.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using Eigen::MatrixXf;
using Eigen::VectorXf;

// ---------------------- SimpleCARFACProcessor ----------------------
// This mirrors your earlier C++ wrapper for CARFAC. It expects the CARFAC C++ API to be available.
// processChunk returns a MatrixXf of shape [channels x time] (same as your Python nap output).
class SimpleCARFACProcessor {
public:
    SimpleCARFACProcessor(int sample_rate = 22050, int num_ears = 1)
        : sample_rate_(sample_rate) {
        CARParams car_params;
        IHCParams ihc_params;
        AGCParams agc_params;

        carfac_ = std::make_unique<CARFAC>(num_ears, sample_rate_, car_params, ihc_params, agc_params);
        num_channels_ = carfac_->num_channels();

        output_ = std::make_unique<CARFACOutput>(true, false, false, false);

        std::cout << "CARFAC initialized: " << num_channels_ << " channels, " << sample_rate_ << " Hz\n";
    }

    // audio_chunk: vector<float> of length N; returns NAP as MatrixXf [channels x time]
    MatrixXf processChunk(const std::vector<float>& audio_chunk) {
        const int N = static_cast<int>(audio_chunk.size());
        Eigen::ArrayXXf segment(1, N);
        for (int i = 0; i < N; ++i) segment(0, i) = audio_chunk[i];

        const bool open_loop = false;
        carfac_->RunSegment(segment, open_loop, output_.get());

        // output_->nap()[0] expected to be [channels x time]
        // convert to Eigen::MatrixXf if needed
        auto &nap_array = output_->nap()[0]; // depending on CARFAC impl this may already be Eigen types
        // If nap_array is ArrayXXf, we can convert:
        MatrixXf nap = nap_array.matrix().cast<float>();
        // Ensure shape is channels x time - if your CARFAC returns time x channels, transpose as needed.
        if (nap.rows() != num_channels_) {
            if (nap.cols() == num_channels_) nap.transposeInPlace();
        }
        return nap;
    }

    int getNumChannels() const { return num_channels_; }
    int getSampleRate() const { return sample_rate_; }

private:
    std::unique_ptr<CARFAC> carfac_;
    std::unique_ptr<CARFACOutput> output_;
    int sample_rate_;
    int num_channels_;
};

// ---------------------- AdvancedSAI (C++ port of your Python) ----------------------
class AdvancedSAI {
public:
    AdvancedSAI(int num_channels = 71, int sai_width = 400, int fs = 22050)
        : num_channels_(num_channels), sai_width_(sai_width), fs_(fs)
    {
        trigger_window_width_ = std::min(400, sai_width_);
        future_lags_ = sai_width_ / 4;
        num_triggers_per_frame_ = 4;

        buffer_width_ = sai_width_ + static_cast<int>((1 + (num_triggers_per_frame_ - 1) / 2.0) * trigger_window_width_);
        if (buffer_width_ < sai_width_) buffer_width_ = sai_width_;

        input_buffer_ = MatrixXf::Zero(num_channels_, buffer_width_);
        output_buffer_ = MatrixXf::Zero(num_channels_, sai_width_);

        // window: sin^2 from approx pi/trigger_window_width to pi
        window_.resize(trigger_window_width_);
        for (int i = 0; i < trigger_window_width_; ++i) {
            double phase = M_PI * (i + 1) / trigger_window_width_;
            window_[i] = static_cast<float>(std::pow(std::sin(phase), 2.0));
        }

        std::cout << "SAI initialized: " << sai_width_ << " lags, " << num_channels_ << " channels\n";
    }

    // nap_segment is expected shape [channels x time]
    MatrixXf process_segment(const MatrixXf& nap_segment) {
        // sanity: if nap_segment shape mismatches channels, try to handle or skip
        if (nap_segment.rows() != num_channels_) {
            std::cerr << "[AdvancedSAI] Warning: nap_segment.rows() != num_channels_ (" 
                      << nap_segment.rows() << " vs " << num_channels_ << ")\n";
            // Optionally try transpose if shape swapped
            if (nap_segment.cols() == num_channels_) {
                MatrixXf t = nap_segment.transpose();
                update_input_buffer(t);
            } else {
                // incompatible shape - skip
                return output_buffer_;
            }
        } else {
            update_input_buffer(nap_segment);
        }
        return compute_sai_frame();
    }

private:
    void update_input_buffer(const MatrixXf& nap_segment) {
        int seg_time = nap_segment.cols();
        if (seg_time >= buffer_width_) {
            // copy last buffer_width_ columns
            for (int ch = 0; ch < num_channels_; ++ch) {
                for (int i = 0; i < buffer_width_; ++i) {
                    input_buffer_(ch, i) = nap_segment(ch, seg_time - buffer_width_ + i);
                }
            }
        } else {
            int shift_amount = seg_time;
            int keep = buffer_width_ - shift_amount;
            // shift left
            for (int ch = 0; ch < num_channels_; ++ch) {
                for (int i = 0; i < keep; ++i) {
                    input_buffer_(ch, i) = input_buffer_(ch, i + shift_amount);
                }
                // copy new segment to tail
                for (int i = 0; i < shift_amount; ++i) {
                    input_buffer_(ch, keep + i) = nap_segment(ch, i);
                }
            }
        }
    }

    MatrixXf compute_sai_frame() {
        output_buffer_.setZero();
        const int num_samples = input_buffer_.cols();
        const int win = trigger_window_width_;
        const int hop = std::max(1, win / 2);

        const int last_window_start = num_samples - win;
        if (last_window_start < 0) return output_buffer_;
        const int first_window_start = last_window_start - (num_triggers_per_frame_ - 1) * hop;

        const int window_range_start = first_window_start - future_lags_;
        const int offset_range_start = first_window_start - sai_width_ + 1;
        if (offset_range_start <= 0 || window_range_start < 0) {
            // not enough history yet
            return output_buffer_;
        }

        // For each channel
        for (int ch = 0; ch < num_channels_; ++ch) {
            // copy channel into std::vector for simple indexing
            std::vector<float> channel_signal(num_samples);
            for (int i = 0; i < num_samples; ++i) channel_signal[i] = input_buffer_(ch, i);

            // smoothing
            std::vector<float> smoothed = channel_signal;
            const float alpha = 0.1f;
            for (int i = 1; i < num_samples; ++i) smoothed[i] = alpha * smoothed[i] + (1.0f - alpha) * smoothed[i - 1];

            for (int trigger_idx = 0; trigger_idx < num_triggers_per_frame_; ++trigger_idx) {
                int window_offset = trigger_idx * hop;
                int current_window_start = window_range_start + window_offset;
                int current_window_end = current_window_start + win;
                if (current_window_end > num_samples) continue;

                // compute windowed trigger
                float peak_value = 0.0f;
                int peak_location = win / 2;
                for (int i = 0; i < win; ++i) {
                    int idx = current_window_start + i;
                    float wv = smoothed[idx] * window_[i];
                    if (wv > peak_value) { peak_value = wv; peak_location = i; }
                }

                int trigger_time = current_window_start + peak_location;
                if (peak_value <= 0.0f) {
                    trigger_time = current_window_start + win / 2;
                    peak_value = 0.1f;
                }

                int correlation_start = trigger_time - sai_width_ + 1 + offset_range_start;
                int correlation_end = correlation_start + sai_width_;
                if (!(correlation_start >= 0 && correlation_end <= num_samples)) {
                    continue;
                }
                // safe to read channel_signal[correlation_start .. correlation_end-1]
                std::vector<float> correlation_segment(sai_width_);
                for (int i = 0; i < sai_width_; ++i) correlation_segment[i] = channel_signal[correlation_start + i];

                float blend = compute_blend_weight(peak_value, trigger_idx);

                // blend to output_buffer_
                for (int lag = 0; lag < sai_width_; ++lag) {
                    // output_buffer_(ch, lag) = (1 - blend) * output_buffer_(ch, lag) + blend * correlation_segment[lag];
                    output_buffer_(ch, lag) = output_buffer_(ch, lag) * (1.0f - blend) + blend * correlation_segment[lag];
                }
            }
        }
        return output_buffer_;
    }

    float compute_blend_weight(float peak_value, int trigger_index) const {
        float base_weight = 0.25f;
        float peak_boost = 2.0f * peak_value / (1.0f + peak_value);
        float temporal_weight = 1.0f - 0.1f * static_cast<float>(trigger_index) / static_cast<float>(num_triggers_per_frame_);
        float final_weight = base_weight * peak_boost * temporal_weight;
        if (final_weight < 0.01f) final_weight = 0.01f;
        if (final_weight > 0.8f) final_weight = 0.8f;
        return final_weight;
    }

    int num_channels_, sai_width_, fs_;
    int trigger_window_width_, future_lags_, num_triggers_per_frame_, buffer_width_;
    MatrixXf input_buffer_, output_buffer_;
    std::vector<float> window_;
};

// ---------------------- Utility: Sine generator + visualization loop ----------------------
std::vector<float> generate_sine(float freq, float duration_s, int sample_rate) {
    int N = static_cast<int>(duration_s * sample_rate);
    std::vector<float> out(N);
    for (int i = 0; i < N; ++i) {
        float t = static_cast<float>(i) / sample_rate;
        out[i] = 0.5f * std::sin(2.0f * M_PI * freq * t);
    }
    return out;
}

int main() {
    const int sample_rate = 22050;
    const int chunk_size = 512;
    const int sai_width = 400;

    // create CARFAC wrapper (requires your CARFAC C++ build)
    SimpleCARFACProcessor carfac(sample_rate);
    const int n_channels = carfac.getNumChannels();

    AdvancedSAI sai(n_channels, sai_width, sample_rate);

    // Visualization setup (OpenCV): heatmap image of [channels x display_width]
    const int display_width = std::min(300, sai_width);
    cv::Mat display_img(n_channels, display_width, CV_8UC3); // color image

    // Simulated "real-time" input: 220 Hz sine for testing
    std::vector<float> test_signal = generate_sine(220.0f, 2.0f, sample_rate); // 2 sec
    size_t idx = 0;
    const double frame_ms = 1000.0 * chunk_size / sample_rate;

    // Rolling SAI buffer for display: keep last display_width columns
    MatrixXf rolling_display = MatrixXf::Zero(n_channels, display_width);

    std::cout << "Starting simulated real-time loop. Press ESC in the display window to quit.\n";

    while (true) {
        // get next chunk (simulate blocking capture)
        std::vector<float> chunk;
        chunk.reserve(chunk_size);
        for (int i = 0; i < chunk_size; ++i) {
            if (idx < test_signal.size()) {
                chunk.push_back(test_signal[idx++]);
            } else {
                // wrap the signal (keep testing)
                idx = 0;
                chunk.push_back(test_signal[idx++]);
            }
        }

        // Process through CARFAC (returns [channels x time])
        MatrixXf nap = carfac.processChunk(chunk);

        // Process through SAI
        MatrixXf sai_frame = sai.process_segment(nap); // [channels x lags]

        // Postprocess for display: simple normalization per channel and power-law
        MatrixXf disp = sai_frame.leftCols(display_width);
        // power-law nonlinearity
        for (int r = 0; r < disp.rows(); ++r) {
            for (int c = 0; c < disp.cols(); ++c) {
                disp(r, c) = std::pow(std::abs(disp(r, c)), 0.75f);
            }
            float maxv = disp.row(r).maxCoeff();
            if (maxv > 1e-9f) disp.row(r) /= maxv;
        }
        // temporal smoothing in rolling buffer
        rolling_display = 0.7f * rolling_display + 0.3f * disp;

        // Convert rolling_display to OpenCV image (grayscale -> apply colormap)
        cv::Mat gray(n_channels, display_width, CV_32F);
        for (int r = 0; r < n_channels; ++r)
            for (int c = 0; c < display_width; ++c)
                gray.at<float>(r, c) = rolling_display(r, c);

        // normalize 0..255
        cv::Mat gray8;
        cv::normalize(gray, gray8, 0, 255, cv::NORM_MINMAX);
        gray8.convertTo(gray8, CV_8U);

        cv::applyColorMap(gray8, display_img, cv::COLORMAP_JET);

        // Show (flip vertically so low freq at bottom if desired)
        cv::Mat to_show;
        cv::flip(display_img, to_show, 0);
        cv::imshow("Realtime SAI (simulated audio)", to_show);

        int key = cv::waitKey(1);
        if (key == 27) break; // ESC

        // simulate real-time delay for chunk
        std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(frame_ms)));
    }

    cv::destroyAllWindows();
    return 0;
}
