from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import argparse
import io
import time
import numpy as np

from PIL import Image
from tflite_runtime.interpreter import Interpreter
from util import set_input_tensor, classify_image, load_test_image, download_model_zoo,parse_options, get_device_arch, get_device_attributes, get_device_type, parse_options, get_cpu_count

model_dir = './mnasnet_1.3_224/'
model_name ='mnasnet_1.3_224.tflite'
repeat=10

model_dir = download_model_zoo(model_dir, model_name)
tflite_model_file = os.path.join(model_dir, model_name)
tflite_model_buf = open(tflite_model_file, "rb").read()
try:
    import tflite
    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
except AttributeError:
    import tflite.Model
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

interpreter = Interpreter(tflite_model_file, num_threads=get_cpu_count())
interpreter.allocate_tensors()

_, height, width, _ = interpreter.get_input_details()[0]['shape']
image = load_test_image('float32', height, width)

numpy_time = np.zeros(repeat)

for i in range(0,repeat):
     start_time = time.time()
     results = classify_image(interpreter, image)

     elapsed_ms = (time.time() - start_time) * 1000
     numpy_time[i] = elapsed_ms

print("tflite %-20s %-19s (%s)" % (model_name, "%.2f ms" % np.mean(numpy_time), "%.2f ms" % np.std(numpy_time)))

