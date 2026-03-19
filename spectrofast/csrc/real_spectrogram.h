int get_number_of_windows(int signal_length, int window_size, int overlap);

int many_real_spectrograms(double *signal, int signal_length,
                            int number_of_signals, int window_size,
                            int overlap, int fft_size, double *spectrogram);
