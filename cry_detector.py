import RPi.GPIO as GPIO
import numpy as np
import librosa as lr
import sounddevice as sd
import tensorflow as tf

# Import model
model = tf.keras.models.load_model('cry_detector.h5')

# preprocessing function, sound device grants a NumPy array, which needs to be made a tensor
def preprocess(audio): 
    #model expects a flat shape
    audio = audio.flatten()

    #normalize fix for bad spectrograms, helps make the amplitude consistent
    audio = audio / (np.reduce_max(np.abs(audio)) + 1e-6)

    #trim or pad to 6 seconds
    audio = audio[:96000]

    if len(audio) < 96000:
        #padding for shorter clips
        zero_padding = np.zeros(96000 - len(audio), dtype=np.float32)
        audio = np.concatenate([zero_padding, audio],0)

    #convert audio to a tensor
    wav = tf.convert_to_tensor(audio, dtype=tf.float32)

    #converts waveform into 20ms windows with 2ms hop
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)

    #drop phase and keep magnitude
    spectrogram = tf.abs(spectrogram)

    #log scaling helps noise become visible in the spectrograms especially the quiet clips
    spectrogram = tf.math.log(spectrogram + 1e-6)

    #channel dimension for CNN (convolutional neural network)
    spectrogram = tf.expand_dims(spectrogram, axis=2) #set channel, 3D tensor
    spectrogram = tf.expand_dims(spectrogram, axis=0) #set batch, 4D tensor
    
    return spectrogram