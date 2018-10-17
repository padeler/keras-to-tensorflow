# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
import numpy as np
import tensorflow as tf
import cv2

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph




def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h%stride==0) else stride - (h % stride) # down
    pad[3] = 0 if (w%stride==0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1,:,:]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:,0:1,:]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1,:,:]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:,-2:-1,:]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad


def preprocess(oriImg, boxsize=368, stride=8, padValue=127):
    scale = float(boxsize) / float(oriImg.shape[0])

    imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    imageToTest_padded, pad = padRightDownCorner(imageToTest, stride, padValue)
    input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels)

    return input_img, pad


def postprocess(output_blobs, model_params, pad, oriImgShape, inputImgShape, hm_idx=0):

    # extract outputs, resize, and remove padding
    heatmap = np.squeeze(output_blobs[hm_idx])  # output at hm_idx is heatmaps
    heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
    heatmap = heatmap[:inputImgShape[0] - pad[2], :inputImgShape[1] - pad[3], :]
    heatmap = cv2.resize(heatmap, (oriImgShape[1], oriImgShape[0]), interpolation=cv2.INTER_CUBIC)

    return heatmap



if __name__ == "__main__":
  input_height = 368
  input_width = 368
  model_file = "./temp/output_graph.pb"
  input_layer = "input_1"
  output_layer = "k2tfout_0"


  graph = load_graph(model_file)

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer

  input_operation = graph.get_operation_by_name(input_name)
  output_operation = graph.get_operation_by_name(output_name)

  # start webcam 
  cap = cv2.VideoCapture(0)
  cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.3)

  k = 0

  with tf.Session(graph=graph) as sess:
    while k!=ord('q'):
      ret, frame = cap.read()
      if not ret:
          raise Exception("VideoCapture.read() returned False")

      frame, pad = preprocess(frame)
      tic = time.time()
      results = sess.run(output_operation.outputs[0], feed_dict={input_operation.outputs[0]: frame})
      dt = time.time() - tic
      print("TTP %.5f, FPS %f" % (dt, 1.0/dt))

      results = np.squeeze(results)
      print("RESULTS: ", results.shape)

      hm = results
      bg = cv2.normalize(hm[:,:,18], None, 0,255, cv2.NORM_MINMAX, cv2.CV_8UC1)
      viz = cv2.normalize(np.sum(hm[:,:,:18],axis=2), None, 0,255, cv2.NORM_MINMAX, cv2.CV_8UC1)
      bg = cv2.resize(bg,(640,480))
      cv2.imshow("BG", bg)

      nose = hm[:,:,0]
      print("nose ", np.min(nose),np.max(nose))
      noseNorm = cv2.normalize(nose,None,0,255,cv2.NORM_MINMAX, cv2.CV_8UC1)

      cv2.imshow("HM", viz)

      k = cv2.waitKey(1)



