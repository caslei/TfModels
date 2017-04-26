# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides utilities for preprocessing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

#用slim简化 ./tensorflow/contrib/slim 文件路径 
slim = tf.contrib.slim


def preprocess_image(image, output_height, output_width, is_training):
  """Preprocesses the given image.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    is_training: `True` if we're preprocessing the image for training and
      `False` otherwise.

  Returns:
    A preprocessed image.
  """
  image = tf.to_float(image)

  # resize_image_with_crop_or_pad(image, target_height, target_width)
  # 通过中心剪切或者填充 0 的方式将图像大小调整到指定的大小
  image = tf.image.resize_image_with_crop_or_pad(image, 
                              output_width, output_height)

  image = tf.subtract(image, 128.0) # 进行减法操作
  image = tf.div(image, 128.0)      # 进行除法操作
  return image
