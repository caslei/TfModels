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
"""Provides utilities to preprocess images.

The preprocessing steps for VGG were introduced in the following technical
report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.ops import control_flow_ops

slim = tf.contrib.slim

# 单下划线开头的变量表示该方法或变量为：私有属性
# “from <模块/包名> import *”，以“_”开头的名称都不会被导入，
# 除非模块或包中的“__all__”列表显式地包含了它们

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 512


def _crop(image, offset_height, offset_width, crop_height, crop_width):
  """Crops the given image using the provided offsets and sizes.

  Note that the method doesn't assume we know the input image size but it does
  assume we know the input image rank.

  Args:
    image: an image of shape [height, width, channels].
    offset_height: a scalar tensor indicating the height offset.
    offset_width: a scalar tensor indicating the width offset.
    crop_height: the height of the cropped image.
    crop_width: the width of the cropped image.

  Returns:
    the cropped (and resized) image.

  Raises:
    InvalidArgumentError: if the rank is not 3 or if the image dimensions are
      less than the crop size.
  """
  original_shape = tf.shape(image) #获取图像的维度：[height, width, channels]

  #-----------------------------------------------------------------------------
  # 断言图像rank的值为3，否则抛出异常信息（图像rank必须为3）， 
  rank_assertion = tf.Assert(tf.equal(tf.rank(image), 3),
                             ['Rank of image must be equal to 3.'])
  # with_dependencies(dependencies, output_tensor, name=None) 
  # 只有所有的dependencies操作运行以后，才运行with_dependencies()语句,并返回 output_tensor
  cropped_shape = control_flow_ops.with_dependencies([rank_assertion],      
          tf.stack([crop_height, crop_width, original_shape[2]])) #返回tf.stack的结果


  # 断言original_shape >= crop_shape,否则返回异常"剪切图像超过原图像"
  size_assertion = tf.Assert( tf.logical_and(     
                    tf.greater_equal(original_shape[0], crop_height),
                    tf.greater_equal(original_shape[1], crop_width)),
                          ['Crop size greater than the image size.'] )
  
  # 将[offset_height, offset_width, 0]堆积为列向量，然后将其转为int32数据类型
  offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

  # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
  # define the crop size.
  # tf.slice(input_, begin, size, name=None):从tensor中提取一张size大小的slice,
  image = control_flow_ops.with_dependencies([size_assertion],
                          tf.slice(image, offsets, cropped_shape))
  #-----------------------------------------------------------------------------
  # tf.Assert()常与tf.control_dependencies()或control_flow_ops.with_dependencies()
  # 一起使用，确保assert_op执行后，才会执行dependencies范围内的其他语句
  #-----------------------------------------------------------------------------


  # 在同一范围内，一旦遇到return语句，函数就直接结算，不在执行同范围内的return后的语句        
  return tf.reshape(image, cropped_shape)


def _random_crop(image_list, crop_height, crop_width):
  """Crops the given list of images.

  The function applies the same crop to each image in the list. This can be
  effectively applied when there are multiple image inputs of the same
  dimension such as:

    image, depths, normals = _random_crop([image, depths, normals], 120, 150)

  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the new height.
    crop_width: the new width.

  Returns:
    the image_list with cropped images.

  Raises:
    ValueError: if there are multiple image inputs provided with different size
      or the images are smaller than the crop dimensions.
  """
  if not image_list:
    raise ValueError('Empty image_list.') # 如果image_list为False则抛出异常

  # Compute the rank assertions.
  rank_assertions = []  # 初始化为空列表

  # 迭代判断image_list中每副图像的 rank => (height,width,channel)
  for i in range(len(image_list)):
    image_rank = tf.rank(image_list[i]) #计算图像的rank

    #断言图像的rank==3，否则抛出异常
    rank_assert = tf.Assert(tf.equal(image_rank, 3),        
        ['Wrong rank for tensor  %s [expected] [actual]', image_list[i].name, 3, image_rank])
    #-----------------------------------------------------------------------------
    # 3和image_rank为何没有 %d 的格式化输出对应， 该语句是否有误 ？？？？？？？？？？？
    #-----------------------------------------------------------------------------

    rank_assertions.append(rank_assert) # 统计总共有多少副图像存在rank不为3的情况

  # rank_assertions[0]成功执行后，返回tf.shape(image_list[0])
  image_shape = control_flow_ops.with_dependencies([rank_assertions[0]], tf.shape(image_list[0]))
      
      
  image_height = image_shape[0]
  image_width  = image_shape[1]

  # 判断剪切尺寸是否同时不超过原图像对应的尺寸
  crop_size_assert = tf.Assert(tf.logical_and(
                        tf.greater_equal(image_height, crop_height),
                        tf.greater_equal(image_width, crop_width)),
                      ['Crop size greater than the image size.'])

  asserts = [rank_assertions[0], crop_size_assert] # 记录 (rank异常 + 剪切异常)
  
  #-----------------------------------------------------------------------------
  # 为什么range 要从 1而不是 0 开始循环 ？？？？？？？？？
  #-----------------------------------------------------------------------------
  for i in range(1, len(image_list)): # [ 1, len(image_list) )
    image = image_list[i]

    asserts.append(rank_assertions[i]) #在asserts后面附加rank_assertions[i]
    shape = control_flow_ops.with_dependencies([rank_assertions[i]], tf.shape(image))
    
    height = shape[0]
    width  = shape[1]

    height_assert = tf.Assert(tf.equal(height, image_height),
      ['Wrong height for tensor %s [expected][actual]',image.name, height, image_height])
    #-----------------------------------------------------------------------------
    # 为什么range 要从 1而不是 0 开始循环 ？？？？？？？？？
    #-----------------------------------------------------------------------------

    width_assert = tf.Assert(tf.equal(width, image_width),
        ['Wrong width for tensor %s [expected][actual]',image.name, width, image_width])

    # list中的append 是将新元素整体添加，extend是将新元素中的每个子对象添加进来
    # [1,2].append([3,4])==>[1,2,[3,4]], [1,2].extend([3,4])==>[1,2,3,4]
    # append adds an element to a list, extend concatenates the first list with another list 
    asserts.extend([height_assert, width_assert])

  # Create a random bounding box.
  #
  # Use tf.random_uniform and not numpy.random.rand as doing the former would
  # generate random numbers at graph eval time, unlike the latter which
  # generates random numbers at graph definition time.
  max_offset_height = control_flow_ops.with_dependencies(
          asserts, tf.reshape(image_height - crop_height + 1, [])) # 返回标量值
  max_offset_width = control_flow_ops.with_dependencies(
          asserts, tf.reshape(image_width - crop_width + 1, []))   # 返回标量值
  #-----------------------------------------------------------------------------
  # 标量scalar 作为tensor时，其维度为:[]
  #-----------------------------------------------------------------------------

  # tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)：
  # 在[minval, maxval)间产生一个均匀分布的随机值，
  # For floats, the default range is [0, 1). For ints, maxval must be specified explicitly.
  offset_height = tf.random_uniform([], maxval=max_offset_height, dtype=tf.int32)
  offset_width  = tf.random_uniform([], maxval=max_offset_width, dtype=tf.int32)

  # 一边循环一边计算的机制，称为生成器（Generator），必须放在[]或者()中
  # 调用单个图像的_crop()方法，通过循环实现大批量图像的切割
  return [_crop(image, offset_height, offset_width, crop_height, crop_width) for image in image_list]



def _central_crop(image_list, crop_height, crop_width):
  """Performs central crops of the given image list.

  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the height of the image following the crop.
    crop_width: the width of the image following the crop.

  Returns:
    the list of cropped images.
  """
  outputs = []
  for image in image_list:  # image_list的维度: [ [height,width],[height,width],....]
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    offset_height = (image_height - crop_height) / 2
    offset_width  = (image_width - crop_width) / 2

    outputs.append(_crop(image, offset_height, offset_width, crop_height, crop_width))
  return outputs


def _mean_image_subtraction(image, means):
  """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """

  # Tensor.get_shape().ndims 返回tensor的 rank, 等价于 tf.rank(tensor) 但后者需sess.run() 求解值.
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')

  num_channels = image.get_shape().as_list()[-1] # 将shape以list的形式返回
  
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  # tf.split(value, num_or_size_splits, axis=0, num=None, name='split')
  # 沿指定axis方向,将张量value切割为num_or_size_splits个子张量
  channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)

  for i in range(num_channels):
    channels[i] -= means[i] #  channels[i] = channels[i] - means[i] 

  # 沿axis的方向将value链接起来 
  #每个channel上的图像张量为[height,width],最后得到[height,width,channels]
  return tf.concat(axis=2, values=channels)


def _smallest_size_at_least(height, width, smallest_side):
  """Computes new shape with the smallest side equal to `smallest_side`.

  Computes new shape with the smallest side equal to `smallest_side` while
  preserving the original aspect ratio.

  Args:
    height: an int32 scalar tensor indicating the current height.
    width: an int32 scalar tensor indicating the current width.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    new_height: an int32 scalar tensor indicating the new height.
    new_width: and int32 scalar tensor indicating the new width.
  """
  #标量转化为张量
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32) 

  height = tf.to_float(height) #标量转化为float张量
  width = tf.to_float(width)

  smallest_side = tf.to_float(smallest_side)

  # tf.cond(pred, fn1, fn2, name=None)： 
  # Return either 'fn1()' or 'fn2()' based on the boolean predicate 'pred'.
  # 类似于c++中的条件表达式 bool ? exp1 : exp2;
  scale = tf.cond(tf.greater(height, width), lambda: smallest_side / width,
                                             lambda: smallest_side / height)

  new_height = tf.to_int32(height * scale)
  new_width  = tf.to_int32(width * scale)

  return new_height, new_width


def _aspect_preserving_resize(image, smallest_side):
  """Resize images preserving the original aspect ratio.

  Args:
    image: A 3-D image `Tensor`.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    resized_image: A 3-D tensor containing the resized image.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  shape = tf.shape(image)
  height = shape[0]
  width = shape[1]
  
  # 调用 _smallest_size_at_least()对图像进行 resize
  new_height, new_width = _smallest_size_at_least(height, width, smallest_side)

  #expand_dims(input, axis=None, name=None, dim=None): 在指定axis上插入维度 “ 1 ” 
  # tf.expand_dims(image, 0) from [height, width, channels] to [1,height, width, channels]
  image = tf.expand_dims(image, 0) 

  # tf.image.resize_bilinear(images, size, align_corners=None, name=None)
  # 利用 bilinear插值，将 images大小调整为 size
  resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                           align_corners=False)

  # tf.squeeze(input, axis=None, name=None, squeeze_dims=None)
  # 在squeeze_dims维度上，将张量中维度值为 1 的维度去掉 [1,2,1，3]--> [2,3]
  resized_image = tf.squeeze(resized_image)
  
  # The None element of the shape corresponds to a variable-sized dimension
  # None 表示shape的具体大小是可变的，只能根据具体情况确定
  resized_image.set_shape([None, None, 3])

  return resized_image


def preprocess_for_train(image,
                         output_height,
                         output_width,
                         resize_side_min=_RESIZE_SIDE_MIN,
                         resize_side_max=_RESIZE_SIDE_MAX):
  """Preprocesses the given image for training.

  Note that the actual resizing scale is sampled from
    [`resize_size_min`, `resize_size_max`].

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    resize_side_min: The lower bound for the smallest side of the image for
      aspect-preserving resizing.
    resize_side_max: The upper bound for the smallest side of the image for
      aspect-preserving resizing.

  Returns:
    A preprocessed image.
  """

  # 在[minval, maxval)之间参数一个维度为[]的随机数，==> 标量
  resize_side = tf.random_uniform([], minval=resize_side_min, 
                                      maxval=resize_side_max+1, dtype=tf.int32)

  # 根据上文的定义，一旦确定选定的计算维度，就以该维度上的比例调整图像大小
  image = _aspect_preserving_resize(image, resize_side)

  # 根据上文， 高度在 [0,image_height-output_height)，间随机取值
  #           宽带在 [0,image_width-output_width)，间随机取值
  image = _random_crop([image], output_height, output_width)[0]

  image.set_shape([output_height, output_width, 3])
  image = tf.to_float(image)

  # tf.image.random_flip_left_right(image, seed=None)
  # Randomly flip an image horizontally (left to right)
  image = tf.image.random_flip_left_right(image)   # 随机水平翻转图像

  return _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])


def preprocess_for_eval(image, output_height, output_width, resize_side):
  """Preprocesses the given image for evaluation.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    resize_side: The smallest side of the image for aspect-preserving resizing.

  Returns:
    A preprocessed image.
  """
  image = _aspect_preserving_resize(image, resize_side)

  # 根据上文定义，_central_crop()的第一个参数应该是一个list： image_list
  # 返回的是另一个image_list，所有通过[0]将图像取出来
  image = _central_crop([image], output_height, output_width)[0]

  image.set_shape([output_height, output_width, 3])
  image = tf.to_float(image)
  
  return _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])


def preprocess_image(image, output_height, output_width, is_training=False,
                     resize_side_min=_RESIZE_SIDE_MIN,
                     resize_side_max=_RESIZE_SIDE_MAX):
  """Preprocesses the given image.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    is_training: `True` if we're preprocessing the image for training and
      `False` otherwise.
    resize_side_min: The lower bound for the smallest side of the image for
      aspect-preserving resizing. If `is_training` is `False`, then this value
      is used for rescaling.
    resize_side_max: The upper bound for the smallest side of the image for
      aspect-preserving resizing. If `is_training` is `False`, this value is
      ignored. Otherwise, the resize side is sampled from
        [resize_size_min, resize_size_max].

  Returns:
    A preprocessed image.
  """
  if is_training:
    return preprocess_for_train(image, output_height, output_width,
                                resize_side_min, resize_side_max)
  else:
    return preprocess_for_eval(image, output_height, output_width,
                               resize_side_min)
