#include <vector>
#include <string>
#include <cmath>
#include <cstdbool>
#include "matplot/matplot.h"


using namespace matplot;

// ---------------- Audio Buffer ----------------
struct AudioBuffer {
    std::mutex mtx;
    std::queue<std::vector<float>> buffer;
    size_t max_size = 10;

    void push(const std::vector<float> &data) {
        std::lock_guard<std::mutex> lock(mtx);
        if (buffer.size() < max_size)
            buffer.push(data);
    }

    bool pop(std::vector<float> &out) {
        std::lock_guard<std::mutex> lock(mtx);
        if (!buffer.empty()) {
            out = buffer.front();
            buffer.pop();
            return true;
        }
        return false;
    }
};

// ---------------- CARFAC Processor ----------------
class CARFACProcessor {
public:
    CARFACProcessor(int n_channels = 71)
        : n_channels(n_channels) {
        state = Eigen::MatrixXf::Zero(n_channels, 1);
    }

    Eigen::MatrixXf process_chunk(const std::vector<float> &audio_chunk) {
        int n_samples = audio_chunk.size();
        Eigen::MatrixXf input(n_channels, n_samples);
        for (int i = 0; i < n_channels; i++)
            for (int j = 0; j < n_samples; j++)
                input(i, j) = audio_chunk[j] * (0.5f + 0.5f * std::sin(i + j*0.01f)); // dummy CARFAC

        // smoothing
        float alpha = 0.1f;
        for (int i = 0; i < n_channels; i++)
            for (int j = 1; j < n_samples; j++)
                input(i, j) = alpha * input(i, j) + (1 - alpha) * input(i, j-1);

        return input;
    }

    int get_n_channels() const { return n_channels; }

private:
    int n_channels;
    Eigen::MatrixXf state;
};

// ---------------- SAI Processor ----------------
class SAIProcessor {
public:
    SAIProcessor(int n_channels = 71, int sai_width = 400)
        : n_channels(n_channels), sai_width(sai_width) {
        buffer_width = sai_width * 2;
        input_buffer = Eigen::MatrixXf::Zero(n_channels, buffer_width);
        output_buffer = Eigen::MatrixXf::Zero(n_channels, sai_width);

        // trigger window
        window.resize(trigger_window_width);
        for (int i = 0; i < trigger_window_width; i++) {
            float pi = 3.1415926f;
            window[i] = std::pow(std::sin(pi / trigger_window_width + i * pi / trigger_window_width), 2.0f);
        }
    }

    Eigen::MatrixXf process_segment(const Eigen::MatrixXf &nap_segment) {
        update_input_buffer(nap_segment);
        output_buffer.setZero();
        int num_samples = input_buffer.cols();
        int window_hop = trigger_window_width / 2;

        int last_window_start = num_samples - trigger_window_width;
        int first_window_start = last_window_start - (num_triggers_per_frame - 1) * window_hop;

        int window_range_start = first_window_start - future_lags;
        int offset_range_start = first_window_start - sai_width + 1;
        if (offset_range_start <= 0 || window_range_start < 0)
            return output_buffer;

        for (int ch = 0; ch < n_channels; ch++) {
            Eigen::VectorXf channel_signal = input_buffer.row(ch);

            for (int trig = 0; trig < num_triggers_per_frame; trig++) {
                int window_start = window_range_start + trig * window_hop;
                int window_end = window_start + trigger_window_width;
                if (window_end > num_samples) continue;

                Eigen::VectorXf trigger_region = channel_signal.segment(window_start, trigger_window_width);
                for (int k = 0; k < trigger_window_width; k++) trigger_region[k] *= window[k];

                float peak_value;
                int peak_index;
                peak_value = trigger_region.maxCoeff(&peak_index);
                int trigger_time = window_start + peak_index;

                int corr_start = trigger_time - sai_width + 1 + offset_range_start;
                int corr_end = corr_start + sai_width;
                if (corr_start >= 0 && corr_end <= num_samples) {
                    Eigen::VectorXf corr_seg = channel_signal.segment(corr_start, sai_width);
                    float blend = compute_blend_weight(peak_value, trig);
                    output_buffer.row(ch) = (1.0f - blend) * output_buffer.row(ch) + blend * corr_seg.transpose();
                }
            }
        }
        return output_buffer;
    }

private:
    int n_channels;
    int sai_width;
    int buffer_width;
    Eigen::MatrixXf input_buffer;
    Eigen::MatrixXf output_buffer;

    int trigger_window_width = 200;
    int num_triggers_per_frame = 4;
    int future_lags = 100;

    std::vector<float> window;

    void update_input_buffer(const Eigen::MatrixXf &nap_segment) {
        int seg_width = nap_segment.cols();
        if (seg_width >= buffer_width)
            input_buffer = nap_segment.rightCols(buffer_width);
        else {
            int shift = seg_width;
            input_buffer.leftCols(buffer_width - shift) = input_buffer.rightCols(buffer_width - shift);
            input_buffer.rightCols(shift) = nap_segment;
        }
    }

    float compute_blend_weight(float peak, int trig_idx) {
        float base = 0.25f;
        float peak_boost = 2.0f * peak / (1.0f + peak);
        float temporal = 1.0f - 0.1f * trig_idx / num_triggers_per_frame;
        float weight = base * peak_boost * temporal;
        if (weight < 0.01f) weight = 0.01f;
        if (weight > 0.8f) weight = 0.8f;
        return weight;
    }
};

// ---------------- Real-time System ----------------
class RealTimeCARFACSAI {
public:
    RealTimeCARFACSAI(int chunk_size = 512, int sample_rate = 22050, int sai_width = 400)
        : chunk_size(chunk_size), sample_rate(sample_rate), sai_width(sai_width),
          carfac(71), sai(71, sai_width), running(false) {}

    void start() {
        running = true;
        audio_thread = std::thread(&RealTimeCARFACSAI::process_audio, this);
        viz_thread = std::thread(&RealTimeCARFACSAI::visualize, this);
    }

    void stop() {
        running = false;
        if (audio_thread.joinable()) audio_thread.join();
        if (viz_thread.joinable()) viz_thread.join();
    }

    void push_audio(const std::vector<float> &audio_chunk) {
        audio_queue.push(audio_chunk);
    }

private:
    int chunk_size;
    int sample_rate;
    int sai_width;

    CARFACProcessor carfac;
    SAIProcessor sai;
    AudioBuffer audio_queue;

    std::atomic<bool> running;
    std::thread audio_thread;
    std::thread viz_thread;

    Eigen::MatrixXf last_sai;

    void process_audio() {
        while (running) {
            std::vector<float> audio_chunk;
            if (audio_queue.pop(audio_chunk)) {
                auto nap_output = carfac.process_chunk(audio_chunk);
                last_sai = sai.process_segment(nap_output);
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }
    }

    void visualize() {
        using namespace matplot;
        int n_channels = carfac.get_n_channels();

        std::vector<std::vector<float>> sai_data(n_channels, std::vector<float>(sai_width, 0.0f));
        auto f = figure(true);
        auto im = imagesc(sai_data);

        while (running) {
            if (last_sai.size() != 0) {
                for (int i = 0; i < n_channels; i++)
                    for (int j = 0; j < sai_width; j++)
                        sai_data[i][j] = last_sai(i, j);

                im->CData(sai_data);
                draw();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
    }
};

// ---------------- Main ----------------
int main() {
    RealTimeCARFACSAI system;
    system.start();

    // Simulate audio input
    for (int i = 0; i < 1000; i++) {
        std::vector<float> audio_chunk(512, 0.1f); // replace with real audio
        system.push_audio(audio_chunk);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    system.stop();
    return 0;
}
