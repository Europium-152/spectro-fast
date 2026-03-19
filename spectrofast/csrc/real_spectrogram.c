#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>
#include <math.h>
#include <string.h>


#ifdef _WIN32
    #include <direct.h>
    #define MKDIR(path) _mkdir(path)
#else
    #include <sys/stat.h>
    #define MKDIR(path) mkdir(path, 0755)
#endif


// Global paths
static char dir_path[4096];
static char wisdom_path[4096];


static void init_wisdom_path() {
    const char *home = getenv("USERPROFILE");
    if (!home) home = getenv("HOME");
    if (!home) return;
    snprintf(dir_path, sizeof(dir_path), "%s/.sfftw", home);
    snprintf(wisdom_path, sizeof(wisdom_path), "%s/.sfftw/wisdom", home);
    MKDIR(dir_path);
}


int get_number_of_windows(int signal_length, int window_size, int overlap) {
    return (signal_length - window_size) / (window_size - overlap) + 1;
}


int many_real_spectrograms(double *signal, int signal_length, int number_of_signals,
                           int window_size, int overlap, double *spectrogram) {

    init_wisdom_path();
    fftw_import_wisdom_from_filename(wisdom_path);

    int input_size = window_size;
    int output_size = window_size / 2 + 1;
    int num_windows = get_number_of_windows(signal_length, window_size, overlap);

    // Allocate FFTW-aligned input and output buffers
    double *in = (double*) fftw_malloc(sizeof(double) * (size_t)signal_length);
    fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (size_t)output_size * (size_t)num_windows);

    if (in == NULL || out == NULL) {
        if (in) fftw_free(in);
        if (out) fftw_free(out);
        return -1;
    }

    // Create batched plan
    fftw_plan p = fftw_plan_many_dft_r2c(1, &input_size, num_windows,
                                          in, NULL,
                                          1, window_size - overlap,
                                          out, NULL,
                                          1, output_size,
                                          FFTW_MEASURE);

    if (p == NULL) {
        fftw_free(in); fftw_free(out);
        return -1;
    }

    fftw_export_wisdom_to_filename(wisdom_path);

    // Process each signal
    for (int k = 0; k < number_of_signals; k++) {
        double *current_signal = &signal[k * signal_length];

        memcpy(in, current_signal, signal_length * sizeof(double));
        fftw_execute(p);

        // Compute magnitude squared
        for (int j = 0; j < (output_size * num_windows); j++) {
            double re = out[j][0];
            double im = out[j][1];
            spectrogram[k * (output_size * num_windows) + j] = re * re + im * im;
        }
    }

    fftw_destroy_plan(p);
    fftw_free(in); fftw_free(out);

    return 0;
}


int many_real_spectrograms_padded(double *signal, int signal_length, int number_of_signals,
                                   int window_size, int overlap, int fft_size, double *spectrogram) {

    init_wisdom_path();
    fftw_import_wisdom_from_filename(wisdom_path);

    int output_size = fft_size / 2 + 1;
    int num_windows = get_number_of_windows(signal_length, window_size, overlap);
    int step = window_size - overlap;

    // Allocate input (fft_size) and output (one FFT) with fftw_malloc
    double *in = (double*) fftw_malloc(sizeof(double) * fft_size);
    fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * output_size);

    if (in == NULL || out == NULL) {
        if (in) fftw_free(in);
        if (out) fftw_free(out);
        return -1;
    }

    // Zero the entire input buffer once — the tail (zero-padding) stays zero
    memset(in, 0, sizeof(double) * fft_size);

    // Plan a single FFT of size fft_size
    fftw_plan p = fftw_plan_dft_r2c_1d(fft_size, in, out, FFTW_MEASURE);

    if (p == NULL) {
        fftw_free(in); fftw_free(out);
        return -1;
    }

    fftw_export_wisdom_to_filename(wisdom_path);

    // Process each signal
    for (int k = 0; k < number_of_signals; k++) {
        double *current_signal = &signal[k * signal_length];

        for (int w = 0; w < num_windows; w++) {
            memcpy(in, &current_signal[w * step], window_size * sizeof(double));

            fftw_execute(p);

            int offset = k * (num_windows * output_size) + w * output_size;
            for (int j = 0; j < output_size; j++) {
                double re = out[j][0];
                double im = out[j][1];
                spectrogram[offset + j] = re * re + im * im;
            }
        }
    }

    fftw_destroy_plan(p);
    fftw_free(in); fftw_free(out);

    return 0;
}
