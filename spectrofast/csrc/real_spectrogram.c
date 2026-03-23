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


// Global paths (non-static: shared with complex_spectrogram.c via concatenation)
char dir_path[4096];
char wisdom_path[4096];


void init_wisdom_path() {
    const char *home = getenv("USERPROFILE");
    if (!home) home = getenv("HOME");
    if (!home) return;
    snprintf(dir_path, sizeof(dir_path), "%s/.sfftw", home);
    snprintf(wisdom_path, sizeof(wisdom_path), "%s/.sfftw/wisdom", home);
    MKDIR(dir_path);
}


// Map integer planner flag to FFTW flag constant
static unsigned int get_fftw_flag(int planner_flag) {
    switch (planner_flag) {
        case 1:  return FFTW_MEASURE;
        case 2:  return FFTW_PATIENT;
        case 3:  return FFTW_EXHAUSTIVE;
        default: return FFTW_ESTIMATE;
    }
}


int get_number_of_windows(int signal_length, int window_size, int overlap) {
    return (signal_length - window_size) / (window_size - overlap) + 1;
}


int many_real_spectrograms(double *signal, int signal_length, int number_of_signals,
                            int window_size, int overlap, int fft_size,
                            int planner_flag, int use_wisdom,
                            double *window, double *spectrogram) {

    if (use_wisdom) {
        init_wisdom_path();
        fftw_import_wisdom_from_filename(wisdom_path);
    }

    unsigned int fftw_flag = get_fftw_flag(planner_flag);

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
    fftw_plan p = fftw_plan_dft_r2c_1d(fft_size, in, out, fftw_flag);

    if (p == NULL) {
        fftw_free(in); fftw_free(out);
        return -1;
    }

    if (use_wisdom) {
        fftw_export_wisdom_to_filename(wisdom_path);
    }

    // Process each signal
    for (int k = 0; k < number_of_signals; k++) {
        double *current_signal = &signal[k * signal_length];

        for (int w = 0; w < num_windows; w++) {
            memcpy(in, &current_signal[w * step], window_size * sizeof(double));

            // Apply window function if provided
            if (window != NULL) {
                for (int i = 0; i < window_size; i++) {
                    in[i] *= window[i];
                }
            }

            fftw_execute(p);

            // Copy complex output directly to spectrogram buffer
            // spectrogram is double* but holds interleaved real/imag pairs
            size_t offset = (size_t)k * num_windows * output_size + (size_t)w * output_size;
            memcpy(&spectrogram[offset * 2], out, output_size * sizeof(fftw_complex));
        }
    }

    fftw_destroy_plan(p);
    fftw_free(in); fftw_free(out);

    return 0;
}
