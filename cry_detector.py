import RPi.GPIO as GPIO
import warnings
warnings.filterwarnings("ignore", module="numpy")
import numpy as np
import pyaudio as pa
import librosa as lr
from tflite_runtime.interpreter import Interpreter
import time
import os, contextlib
import threading

@contextlib.contextmanager
def suppress_logs():
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(devnull)

def record():
    CHUNK = 1024
    FORMAT = pa.paInt16
    CHANNELS = 1
    RATE = 16000

    print("Recording...")

    with suppress_logs():
        p = pa.PyAudio()
        stream = p.open(format = FORMAT,
                        channels = CHANNELS,
                        rate = RATE,
                        input = True,
                        frames_per_buffer = CHUNK)

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

def cleanup():
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.LOW)
    GPIO.output(in3, GPIO.LOW)
    GPIO.output(in4, GPIO.LOW)
    GPIO.cleanup()

def setup():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(in1, GPIO.OUT)
    GPIO.setup(in2, GPIO.OUT)
    GPIO.setup(in3, GPIO.OUT)
    GPIO.setup(in4, GPIO.OUT)

    #initializing
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.LOW)
    GPIO.output(in3, GPIO.LOW)
    GPIO.output(in4, GPIO.LOW)

def motor_spin():
    global in1, in2, in3, in4, step_sleep, direction, motor_pins, motor_step_counter, running

    while(1):

        motor_pins = [in1, in2, in3, in4]
        motor_step_counter = 0

        while running:
            for pin in range(0, len(motor_pins)):
                GPIO.output(motor_pins[pin], step_sequence[motor_step_counter][pin])
            if direction==True:
                motor_step_counter = (motor_step_counter - 1) % 8
            elif direction==False:
                motor_step_counter = (motor_step_counter + 1) % 8
            else:
                print("Direction Error")
                cleanup()
                exit(1)
            time.sleep(step_sleep)

def run():
    global counter, glob_counter, running, max_highs
    setup()

    motor_thread = threading.Thread(target=motor_spin, daemon=True).start()

    try:
        while (1):
            audio = record()

            with suppress_logs():
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

            if (pred_num >= 0.60): #Prediction number, closer to 1 means more accurate set off logic when a close number detected
                running = True
                counter += 1 #increase counters
                glob_counter += 1
            else:
                running = False
                counter = 0 #reset in a row counter

    except KeyboardInterrupt:
        cleanup()
        print("\nHighs in a row: " + str(counter))
        print("Total Highs: " + str(glob_counter))

counter = 0
glob_counter = 0
max_highs = 0

in1 = 17
in2 = 18
in3 = 27
in4 = 22

step_sleep = 0.002

direction = False

step_sequence = [[1,0,0,1],
                [1,0,0,0],
                [1,1,0,0],
                [0,1,0,0],
                [0,1,1,0],
                [0,0,1,0],
                [0,0,1,1],
                [0,0,0,1]]
running = False

run()