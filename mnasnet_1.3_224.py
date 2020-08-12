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
from util import set_input_tensor, classify_image, load_test_image, download_model_zoo,parse_options, get_device_arch, get_device_attributes, get_device_type, parse_options

model_dir = './mnasnet_1.3_224/'
model_name ='mnasnet_1.3_224.tflite'
repeat=10

interpreter = Interpreter(tflite_model_file)
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

