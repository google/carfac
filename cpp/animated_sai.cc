#include <iostream>
#include <vector>
#include <memory>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <iomanip>

#include <Eigen/Core>

// Include your CARFAC/SAI headers
#include "carfac.h"
#include "sai.h"
#include "car.h"
#include "agc.h"
#include "ihc.h"
#include "common.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 便宜上：Eigen の型エイリアス（あなたの環境で既に typedef 済みなら不要）
using Eigen::ArrayXXf;
using Eigen::ArrayXf;
using ArrayXX = ArrayXXf;
using ArrayX  = ArrayXf;

class SimpleCARFACProcessor {
public:
    SimpleCARFACProcessor(int sample_rate = 22050, int num_ears = 1) 
        : sample_rate_(sample_rate) {
        CARParams car_params;
        IHCParams ihc_params;
        AGCParams agc_params;

        carfac_ = std::make_unique<CARFAC>(num_ears, sample_rate, car_params, ihc_params, agc_params);
        num_channels_ = carfac_->num_channels();

        // NAP だけ使う
        output_ = std::make_unique<CARFACOutput>(true, false, false, false);

        std::cout << "CARFAC initialized: " << num_channels_ << " channels, " 
                  << sample_rate << " Hz" << std::endl;
    }
    
    // NAP を [channels x time] で返す（※転置しない）
    ArrayXX processChunk(const std::vector<float>& audio_chunk) {
        ArrayXX segment(1, static_cast<int>(audio_chunk.size()));
        for (size_t i = 0; i < audio_chunk.size(); ++i) segment(0, static_cast<int>(i)) = audio_chunk[i];

        const bool open_loop = false;
        carfac_->RunSegment(segment, open_loop, output_.get());

        // ここで output_->nap()[0] は [channels x time] である前提
        return output_->nap()[0];
    }
    
    int getNumChannels() const { return num_channels_; }
    int getSampleRate() const { return sample_rate_; }
    
private:
    std::unique_ptr<CARFAC> carfac_;
    std::unique_ptr<CARFACOutput> output_;
    int sample_rate_;
    int num_channels_;
};

class SimpleSAI {
public:
    SimpleSAI(int num_channels = 71, int sai_width = 400, int sample_rate = 22050)
        : num_channels_(num_channels), sai_width_(sai_width), sample_rate_(sample_rate) {
        
        trigger_window_width_ = std::min(400, sai_width_);
        future_lags_ = sai_width_ / 4;           // 使うが過度には依存しない
        num_triggers_per_frame_ = 4;
        
        buffer_width_ = sai_width_ + static_cast<int>((1 + (num_triggers_per_frame_ - 1) / 2.0) * trigger_window_width_);
        if (buffer_width_ < sai_width_) buffer_width_ = sai_width_;

        input_buffer_.resize(num_channels_, buffer_width_);
        input_buffer_.setZero();

        output_buffer_.resize(num_channels_, sai_width_);
        output_buffer_.setZero();
        
        // 窓関数（sine^2）
        window_.resize(trigger_window_width_);
        for (int i = 0; i < trigger_window_width_; ++i) {
            double phase = M_PI * (i + 1) / trigger_window_width_;
            window_[i] = static_cast<float>(std::pow(std::sin(phase), 2.0));
        }
        
        std::cout << "SAI initialized: " << sai_width_ << " lags, " 
                  << num_channels_ << " channels" << std::endl;
    }
    
    ArrayXX processSegment(const ArrayXX& nap_segment) {
        // 期待する形は [channels x time]
        if (nap_segment.rows() != num_channels_) {
            // rows と num_channels が食い違う場合は安全にスキップ
            // 必要ならここで transpose 検知して nap_segment.transpose() を使う分岐を入れてもOK
            return output_buffer_;
        }
        updateInputBuffer(nap_segment);
        return computeSAIFrame();
    }
    
private:
    void updateInputBuffer(const ArrayXX& nap_segment) {
        const int seg_time = nap_segment.cols();     // time 次元（サンプル数）
        const int copy_time = std::min(seg_time, buffer_width_);

        if (seg_time >= buffer_width_) {
            // 右側（最新）の buffer_width_ サンプルをコピー
            for (int ch = 0; ch < num_channels_; ++ch) {
                for (int i = 0; i < buffer_width_; ++i) {
                    input_buffer_(ch, i) = nap_segment(ch, seg_time - buffer_width_ + i);
                }
            }
        } else {
            // 左へシフト → 末尾に新データ
            const int shift = seg_time;
            const int keep  = buffer_width_ - shift;
            for (int ch = 0; ch < num_channels_; ++ch) {
                // 左に詰める（古い先頭が落ちる）
                for (int i = 0; i < keep; ++i) {
                    input_buffer_(ch, i) = input_buffer_(ch, i + shift);
                }
                // 末尾に追記
                for (int i = 0; i < shift; ++i) {
                    input_buffer_(ch, keep + i) = nap_segment(ch, i);
                }
            }
        }
    }
    
    ArrayXX computeSAIFrame() {
        output_buffer_.setZero();
        
        const int num_samples = input_buffer_.cols();
        const int win = trigger_window_width_;
        const int hop = std::max(1, win / 2);

        // 最新の窓の開始位置
        const int last_window_start = num_samples - win;
        if (last_window_start < 0) return output_buffer_;

        // 最初の窓の開始位置（未来ラグ分だけ少し前から）
        const int first_window_start = std::max(0, last_window_start - (num_triggers_per_frame_ - 1) * hop - future_lags_);
        
        for (int ch = 0; ch < num_channels_; ++ch) {
            // 1チャンネル分のコピー
            std::vector<float> x(num_samples);
            for (int i = 0; i < num_samples; ++i) x[i] = input_buffer_(ch, i);

            // 簡易スムージング（ピークトリガ検出を安定化）
            std::vector<float> s(x);
            const float alpha = 0.1f;
            for (int i = 1; i < num_samples; ++i) s[i] = alpha * s[i] + (1.0f - alpha) * s[i - 1];

            for (int trig = 0; trig < num_triggers_per_frame_; ++trig) {
                const int current_window_start = first_window_start + trig * hop;
                const int current_window_end   = current_window_start + win;
                if (current_window_end > num_samples) break;

                // 窓内ピーク検出（windowed）
                float peak_val = 0.0f;
                int peak_loc = win / 2;
                for (int i = 0; i < win; ++i) {
                    const int idx = current_window_start + i;
                    const float wv = s[idx] * window_[i];
                    if (wv > peak_val) { peak_val = wv; peak_loc = i; }
                }
                int trigger_time = current_window_start + peak_loc;
                if (peak_val <= 0.0f) {
                    trigger_time = current_window_start + win / 2;
                    peak_val = 0.1f;
                }

                const float blend = computeBlendWeight(peak_val, trig);

                // ここがコア：lag ごとに「trigger_time - lag」を参照
                for (int lag = 0; lag < sai_width_; ++lag) {
                    const int idx = trigger_time - lag;        // 過去方向へ
                    if (idx < 0 || idx >= num_samples) break;  // これ以上は範囲外
                    // 出力は [channels x lags]
                    // ここでの lags は 0..(sai_width_-1) で、idx が負になったら終了
                    output_buffer_(ch, lag) = (1.0f - blend) * output_buffer_(ch, lag) + blend * x[idx];
                }
            }
        }
        return output_buffer_;
    }
    
    float computeBlendWeight(float peak_value, int trigger_index) const {
        const float base = 0.25f;
        const float peak_boost = 2.0f * peak_value / (1.0f + peak_value);
        const float temporal   = 1.0f - 0.1f * trigger_index / std::max(1, num_triggers_per_frame_);
        float w = base * peak_boost * temporal;
        if (w < 0.01f) w = 0.01f;
        if (w > 0.80f) w = 0.80f;
        return w;
    }
    
    int num_channels_, sai_width_, sample_rate_;
    int trigger_window_width_, future_lags_, num_triggers_per_frame_, buffer_width_;
    ArrayXX input_buffer_, output_buffer_;
    std::vector<float> window_;
};

// ==== Utility ====

// 単純サイン波
std::vector<float> generateSineWave(float frequency, float duration, int sample_rate) {
    const int N = static_cast<int>(duration * sample_rate);
    std::vector<float> y(N);
    for (int i = 0; i < N; ++i) {
        float t = static_cast<float>(i) / sample_rate;
        y[i] = 0.5f * std::sin(2.0f * static_cast<float>(M_PI) * frequency * t);
    }
    return y;
}

// ピッチ解析（チャネル平均の自己相関に相当する指標）
std::string analyzePitch(const ArrayXX& sai_data, int sample_rate) {
    if (sai_data.size() == 0) return "No data";

    // チャネル方向に平均 → [1 x lags]
    ArrayX mean_lag = sai_data.colwise().mean();

    // 0 〜 2ms は無視（ゼロラグ付近の支配を避ける）
    const int min_lag = std::max(1, static_cast<int>(0.002f * sample_rate)); // ~2ms
    int max_idx = min_lag;
    float max_val = -1e9f;
    for (int i = min_lag; i < mean_lag.size(); ++i) {
        if (mean_lag(i) > max_val) { max_val = mean_lag(i); max_idx = i; }
    }

    if (max_val > 0.3f && max_idx > 0) {
        float f0 = static_cast<float>(sample_rate) / static_cast<float>(max_idx);
        return "Detected pitch: " + std::to_string(static_cast<int>(std::round(f0))) + " Hz";
    }
    return "No clear pitch detected";
}

void printSAISummary(const ArrayXX& sai_data, int frame_num) {
    std::cout << "Frame " << frame_num << " - SAI dimensions: " 
              << sai_data.rows() << "x" << sai_data.cols() << std::endl;
    const float max_val = sai_data.maxCoeff();
    const float mean_val = sai_data.mean();
    std::cout << "  Max correlation: " << std::fixed << std::setprecision(3) << max_val 
              << ", Mean: " << mean_val << std::endl;
}

void saveSAIToFile(const ArrayXX& sai_data, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filename << std::endl;
        return;
    }
    for (int ch = 0; ch < sai_data.rows(); ++ch) {
        for (int lag = 0; lag < sai_data.cols(); ++lag) {
            file << sai_data(ch, lag);
            if (lag < sai_data.cols() - 1) file << ",";
        }
        file << "\n";
    }
    file.close();
    std::cout << "SAI data saved to: " << filename << std::endl;
}

int main() {
    std::cout << "Console-based CARFAC-SAI Processor" << std::endl;
    std::cout << "==================================" << std::endl;

    try {
        const int sample_rate = 22050;
        const int sai_width   = 400;

        SimpleCARFACProcessor carfac_processor(sample_rate);
        const int num_channels = carfac_processor.getNumChannels();

        SimpleSAI sai_processor(num_channels, sai_width, sample_rate);

        // テスト周波数
        std::vector<float> test_frequencies = {220.0f, 440.0f, 880.0f};

        for (float freq : test_frequencies) {
            std::cout << "\nTesting with " << freq << " Hz sine wave:" << std::endl;
            std::vector<float> test_signal = generateSineWave(freq, 1.0f, sample_rate);

            const int chunk_size = 512;
            int frame_count = 0;

            for (size_t i = 0; i < test_signal.size(); i += chunk_size) {
                const size_t end_idx = std::min(i + chunk_size, test_signal.size());
                std::vector<float> chunk(test_signal.begin() + i, test_signal.begin() + end_idx);

                ArrayXX nap = carfac_processor.processChunk(chunk); // [channels x time]
                ArrayXX sai_frame = sai_processor.processSegment(nap); // [channels x lags]

                ++frame_count;
                if (frame_count % 10 == 0) {
                    printSAISummary(sai_frame, frame_count);
                    std::cout << "  " << analyzePitch(sai_frame, sample_rate) << std::endl;
                }

                if (end_idx == test_signal.size()) {
                    std::string filename = "sai_output_" + std::to_string(static_cast<int>(freq)) + "Hz.csv";
                    saveSAIToFile(sai_frame, filename);
                }
            }
        }

        std::cout << "\nProcessing complete! Check the generated CSV files for SAI data." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
