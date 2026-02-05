import numpy as np
from tflite_runtime.interpreter import Interpreter

#Load TFlite model and allocate tensors
interpreter = Interpreter(model_path="cry_detector.tflite")
interpreter.allocate_tensors()

#Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#Test model on random input data
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

#Function get_tensor() returns copy of tensor data
#Use tensor() in order to get a pointer to the tensor
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)