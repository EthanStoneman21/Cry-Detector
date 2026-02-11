import RPi.GPIO as GPIO
import numpy as np
import pyaudio as pa
import librosa as lr
from tflite_runtime.interpreter import Interpreter
import time

def record():
    CHUNK = 1024
    FORMAT = pa.paInt16
    CHANNELS = 1
    RATE = 16000

    p = pa.PyAudio()

    stream = p.open(format = FORMAT,
                    channels = CHANNELS,
                    rate = RATE,
                    input = True,
                    frames_per_buffer = CHUNK)
    print("Recording...")

    frames = []
    seconds = 6
    for i in range(0, int(RATE / CHUNK * seconds)):
        data = stream.read(CHUNK)
        num_arr = np.frombuffer(data, dtype = np.int16)
        frames.append(num_arr)

    print("Recording stopped")

    stream.stop_stream()
    stream.close()
    p.terminate()

    return np.concatenate(frames) #convert to numpy

# preprocessing function, pyaudio grants raw audio, need a NumPy array, take preprocessing function from training and convert to numpy for tflite
def preprocess(audio): 
    #convert audio to float32
    audio = audio.astype(np.float32)

    #model expects a flat shape
    audio = audio.flatten()

    #normalize fix for bad spectrograms, helps make the amplitude consistent
    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    #trim or pad to 6 seconds
    audio = audio[:96000]

    if len(audio) < 96000:
        #padding for shorter clips
        audio = np.concatenate([audio, np.zeros(96000 - len(audio), dtype = np.float32)])

    #converts waveform into 20ms windows with 2ms hop
    spectrogram = lr.stft(audio, n_fft = 512, hop_length = 32, center = True, pad_mode = "reflect")

    #drop phase and keep magnitude
    spectrogram = np.abs(spectrogram)

    #log scaling helps noise become visible in the spectrograms especially the quiet clips
    spectrogram = np.log(spectrogram + 1e-6)

    spectrogram = spectrogram.T
    spectrogram = spectrogram[:2991, :257]

    #channel dimension for CNN (convolutional neural network)
    spectrogram = np.expand_dims(spectrogram, axis = 2) #set channel, 3D tensor
    spectrogram = np.expand_dims(spectrogram, axis = 0) #set batch, 4D tensor
    
    return spectrogram.astype(np.float32)

def run():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(17, GPIO.OUT)

    while (1):
        audio = record()

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
        pred_num = float(output_data.squeeze()) #Extract just the prediction number to use for pi
        print(pred_num)

        if (pred_num >= 0.55): #Prediction number, closer to 1 means more accurate set off logic when a close number detected
            GPIO.output(17, GPIO.HIGH)
        else:
            GPIO.output(17, GPIO.LOW)

        exit = input("Would you like to continue? (y or n) ")

        if exit == "n" or exit == "N":
            GPIO.cleanup()
            return
run()
