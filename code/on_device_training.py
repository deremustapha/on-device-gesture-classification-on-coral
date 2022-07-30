import argparse
import time
import os
import pathlib
from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.adapters import classify
from scipy.io import loadmat
import numpy as np
from pycoral.learn.imprinting.engine import ImprintingEngine
from pycoral.utils.edgetpu import make_interpreter

dataset = loadmat('test_data')
data = dataset['data']
label = dataset['label']
model_file = os.path.join('QAT_BMIS_EMG_NET_edgetpu.tflite')


engine = ImprintingEngine(model_file)# keep classes
extractor = make_interpreter(engine.serialize_extractor_model(), device=':0')
extractor.allocate_tensors()
shape = common.input_size(extractor)


print('----------------      Start training     -----------------')
num_classes = engine.num_classes
for class_id, tensors in enumerate(data):
    for tensor in tensors:
        common.set_input(extractor, tensor)
        extractor.invoke()
        embedding = classify.get_scores(extractor)
        engine.train(embedding, class_id=num_classes + class_id)
print('----------------     Training finished!  -----------------')
