/* complex_spectrogram.c
   This file is concatenated after real_spectrogram.c by the cffi build script,
   so all includes, init_wisdom_path(), get_number_of_windows(), get_fftw_flag(),
   and wisdom_path are already available. Do NOT add #include directives here. */


int many_complex_spectrograms(double *signal, int signal_length, int number_of_signals,
                              int window_size, int overlap, int fft_size,
                              int planner_flag, int use_wisdom,
                              double *window, double *spectrogram) {

    if (use_wisdom) {
        init_wisdom_path();
        fftw_import_wisdom_from_filename(wisdom_path);
    }

    unsigned int fftw_flag = get_fftw_flag(planner_flag);

    int output_size = fft_size;  /* complex FFT: full spectrum */
    int num_windows = get_number_of_windows(signal_length, window_size, overlap);
    int step = window_size - overlap;

    /* Allocate FFTW-aligned complex input and output buffers */
    fftw_complex *in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_size);
    fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_size);

    if (in == NULL || out == NULL) {
        if (in) fftw_free(in);
        if (out) fftw_free(out);
        return -1;
    }

    /* Zero the entire input buffer once — the tail (zero-padding) stays zero */
    memset(in, 0, sizeof(fftw_complex) * fft_size);

    /* Plan a single complex-to-complex FFT of size fft_size */
    fftw_plan p = fftw_plan_dft_1d(fft_size, in, out, FFTW_FORWARD, fftw_flag);

    if (p == NULL) {
        fftw_free(in); fftw_free(out);
        return -1;
    }

    if (use_wisdom) {
        fftw_export_wisdom_to_filename(wisdom_path);
    }

    /* Process each signal.
       signal is double* pointing to interleaved real/imag pairs (numpy complex128).
       Each complex sample is 2 doubles, so signal_length complex samples = 2*signal_length doubles.
       We cast to fftw_complex* for convenient indexing. */
    for (int k = 0; k < number_of_signals; k++) {
        fftw_complex *current_signal = (fftw_complex*) &signal[k * signal_length * 2];

        for (int w = 0; w < num_windows; w++) {
            memcpy(in, &current_signal[w * step], window_size * sizeof(fftw_complex));

            /* Apply window function if provided.
               Window is real-valued (double*), so multiply both real and imag parts. */
            if (window != NULL) {
                for (int i = 0; i < window_size; i++) {
                    in[i][0] *= window[i];
                    in[i][1] *= window[i];
                }
            }

            fftw_execute(p);

            /* Copy complex output directly to spectrogram buffer
               spectrogram is double* but holds interleaved real/imag pairs */
            size_t offset = (size_t)k * num_windows * output_size + (size_t)w * output_size;
            memcpy(&spectrogram[offset * 2], out, output_size * sizeof(fftw_complex));
        }
    }

    fftw_destroy_plan(p);
    fftw_free(in); fftw_free(out);

    return 0;
}
