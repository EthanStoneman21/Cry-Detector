import RPi.GPIO as GPIO
import numpy as np
import librosa as lr
import pyaudio as ad
from tflite_runtime.interpreter import Interpreter

# preprocessing function, pyaudio grants raw audio,, need a NumPy array
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

    #converts waveform into 20ms windows with 2ms hop using librosa
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)

    #drop phase and keep magnitude
    spectrogram = np.abs(spectrogram)

    #log scaling helps noise become visible in the spectrograms especially the quiet clips
    spectrogram = np.math.log(spectrogram + 1e-6)

    #channel dimension for CNN (convolutional neural network)
    spectrogram = np.expand_dims(spectrogram, axis=2) #set channel, 3D tensor
    spectrogram = np.expand_dims(spectrogram, axis=0) #set batch, 4D tensor
    
    return spectrogram.astype(np.float32)

#Load TFlite model and allocate tensors
interpreter = Interpreter(model_path="cry_detector.tflite")
interpreter.allocate_tensors()

#Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

spectrogram = preprocess(audio)
interpreter.set_tensor(input_details[0]['index'], spectrogram)

interpreter.invoke()

#Function get_tensor() returns copy of tensor data
#Use tensor() in order to get a pointer to the tensor
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)

input