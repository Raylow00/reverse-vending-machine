# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

import argparse
import sys
import time
import os
import cv2
from image_classifier import ImageClassifier
from image_classifier import ImageClassifierOptions

# Visualization parameters
_ROW_SIZE = 20  # pixels
_LEFT_MARGIN = 24  # pixels
_TEXT_COLOR = (0, 0, 255)  # red
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_FPS_AVERAGE_FRAME_COUNT = 10


def run(model: str, max_results: int, num_threads: int, enable_edgetpu: bool,
        camera_id: int, width: int, height: int) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
      model: Name of the TFLite image classification model.
      max_results: Max of classification results.
      num_threads: Number of CPU threads to run the model.
      enable_edgetpu: Whether to run the model on EdgeTPU.
      camera_id: The camera id to be passed to OpenCV.
      width: The width of the frame captured from the camera.
      height: The height of the frame captured from the camera.
  """

  # Initialize the image classification model
  options = ImageClassifierOptions(
      num_threads=num_threads,
      max_results=max_results,
      enable_edgetpu=enable_edgetpu)
  classifier = ImageClassifier(model, options)

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Capture image from camera
  os.system("fswebcam /home/pi/Pictures/image.jpg")

  # Test image
  # image_path = "/home/pi/RVM/Lobe/tflite_model/can.jpg"
  image_path = "/home/pi/Pictures/image.jpg"
  image = cv2.imread(image_path)
  image = cv2.flip(image, 1)    # Optional preprocessing step
  # List classification results
  categories, final_prediction = classifier.classify(image)
  # Show classification results on the image
  for idx, category in enumerate(categories):
    class_name = category.label
    score = round(category.score, 2)
    result_text = class_name + ' (' + str(score) + ')'
    print("Prediction: {}, Probability: {}".format(class_name, str(score)))
  
  print("Final prediction: ", final_prediction)

  # Calculate the FPS
  if counter % _FPS_AVERAGE_FRAME_COUNT == 0:
    end_time = time.time()
    fps = _FPS_AVERAGE_FRAME_COUNT / (end_time - start_time)
    start_time = time.time()

  # Show the FPS
  fps_text = 'FPS = ' + str(int(fps)) + ' second(s)'
  print(fps_text)


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Name of image classification model.',
      required=False,
      default='efficientnet_lite0.tflite')
  parser.add_argument(
      '--maxResults',
      help='Max of classification results.',
      required=False,
      default=3)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      default=480)
  args = parser.parse_args()

  run(args.model, int(args.maxResults), int(args.numThreads),
      bool(args.enableEdgeTPU), int(args.cameraId), args.frameWidth,
      args.frameHeight)


if __name__ == '__main__':
  main()
