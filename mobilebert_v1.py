from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tflite_runtime.interpreter import Interpreter
import os
import pdb
import argparse
import io
import time
import numpy as np

from PIL import Image
from util import set_input_tensor, classify_image, load_test_image, download_model_zoo,parse_options, get_device_arch, get_device_attributes, get_device_type, parse_options, get_cpu_count

def set_input_tensors(interpreter, inputs, token_types, valid_length, input_shape ):
  #pdb.set_trace()
  input_details = interpreter.get_input_details()
  interpreter.set_tensor(input_details[0]['index'], inputs)


def classify_data(interpreter, inputs, token_types, valid_length, input_shape, top_k=1):
  """Returns a sorted array of classification results."""
  set_input_tensors(interpreter, inputs, token_types, valid_length, input_shape)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]

#pdb.set_trace()
model_dir = './mobilebert_v1/'
model_name ='mobilebert_1_default_1.tflite'
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

input_dtype="float32"
input_dtype="int32"
batch = 1
seq_length = 384

#Random inputs for BERT network. Previous example had segment_ids (valid_length) as just (batch,) which was causing errors. 
inputs = np.random.randint(0, 2000, size=(batch, seq_length)).astype(input_dtype)
token_types = np.random.uniform(size=(batch, seq_length)).astype(input_dtype)
valid_length = np.random.uniform(size=(batch, seq_length)).astype(input_dtype)

input_tensors = ['input_ids', 'input_mask', 'segment_ids']
input_shape = (batch, seq_length)

#_, height, width, _ = interpreter.get_input_details()[0]['shape']


numpy_time = np.zeros(repeat)

for i in range(0,repeat):
     start_time = time.time()
     results = classify_data(interpreter, inputs, token_types, valid_length, input_shape)

     elapsed_ms = (time.time() - start_time) * 1000
     numpy_time[i] = elapsed_ms

print("tflite %-20s %-19s (%s)" % (model_name, "%.2f ms" % np.mean(numpy_time), "%.2f ms" % np.std(numpy_time)))

