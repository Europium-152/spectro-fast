/* complex_spectrogram.c
   This file is concatenated after real_spectrogram.c by the cffi build script,
   so all includes, init_wisdom_path(), get_number_of_windows(), and wisdom_path
   are already available. Do NOT add #include directives here. */


int many_complex_spectrograms(double *signal, int signal_length, int number_of_signals,
                              int window_size, int overlap, int fft_size, double *spectrogram) {

    init_wisdom_path();
    fftw_import_wisdom_from_filename(wisdom_path);

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
    fftw_plan p = fftw_plan_dft_1d(fft_size, in, out, FFTW_FORWARD, FFTW_MEASURE);

    if (p == NULL) {
        fftw_free(in); fftw_free(out);
        return -1;
    }

    fftw_export_wisdom_to_filename(wisdom_path);

    /* Process each signal.
       signal is double* pointing to interleaved real/imag pairs (numpy complex128).
       Each complex sample is 2 doubles, so signal_length complex samples = 2*signal_length doubles.
       We cast to fftw_complex* for convenient indexing. */
    for (int k = 0; k < number_of_signals; k++) {
        fftw_complex *current_signal = (fftw_complex*) &signal[k * signal_length * 2];

        for (int w = 0; w < num_windows; w++) {
            memcpy(in, &current_signal[w * step], window_size * sizeof(fftw_complex));

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
