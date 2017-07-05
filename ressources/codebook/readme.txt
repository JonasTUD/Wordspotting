Das Codebuch ist als binäre Datei (4 byte float) abgelegt. Es enthält 4096
Visual Words mit je 128 Dimensionen (Dimensionalität des SIFT Deskriptors).
Laden Sie es wie folgt:

input_file = open('codebook.bin', 'r')
codebook = np.fromfile(input_file, dtype='float32')
codebook = np.reshape(codebook, (4096,128))
