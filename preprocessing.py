import matplotlib.pyplot as plt
import numpy as np
import librosa

def create_spectrogram(audio_path):
    '''
    Creates and saves spectrogram from given audio file (.wav)
        Spectrogram will be saved to the same file path but as an image (.png)
        Using 128 mel-spectrogram bins
    audio_path: path to wav file to make spectrogram from
    '''

    # get spectrogram
    y, sr = librosa.load(audio_path)
    S = librosa.feature.melspectrogram(y=y, sr=sr, fmax=(sr/2.0), n_mels=128)

    # plot
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=(sr/2.0), ax=ax)
    # TODO: remove axes and ticks and all that so it's just the image?
    # https://stackoverflow.com/questions/8218608/savefig-without-frames-axes-only-content

    # save as image
    # TODO
