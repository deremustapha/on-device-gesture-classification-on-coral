import argparse
import time
import os
import pathlib
from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.adapters import classify
from scipy.io import loadmat
import numpy as np

dataset = loadmat('stft_test_data.mat')
data = dataset['data']
label = dataset['label']
model_file = os.path.join('QAT_BMIS_EMG_NET_STFT_INPUT_edgetpu.tflite')

# Initialize the TF interpreter
interpreter = edgetpu.make_interpreter(model_file)
interpreter.allocate_tensors()
size = common.input_size(interpreter)
# Run an inference
num_samples = 20
total = data.shape[0]
predictions = np.empty_like(label)
print("Real-time Motor Intent Inference on Coral TPU")

for i in range(num_samples):
    start = time.perf_counter()
    common.set_input(interpreter, data[i])
    interpreter.invoke()
    inference_time = time.perf_counter() - start
    gesture = classify.get_classes(interpreter, top_k=1)
    gnd_truth = label[i]
    print('%.1fms' % (inference_time * 1000))

    for c in gesture:
        print('On-device Inference: {}  True Label {} '.format(c.id, gnd_truth))
        print('Confidence score is {}'.format(c.score))

for i in range(total):

    common.set_input(interpreter, data[i])
    interpreter.invoke()
    gesture = classify.get_classes(interpreter, top_k=1)
    for c in gesture:
        predictions[i] = c.id

correct = np.sum(predictions == label)
print("Overall accuracy: {}".format((correct / total)*100))
