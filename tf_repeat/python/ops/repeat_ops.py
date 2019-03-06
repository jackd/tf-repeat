# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Use repeat ops in python."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import array_ops
from tensorflow.python.framework import load_library
from tensorflow.python.framework import ops
from tensorflow.python.platform import resource_loader

filename = resource_loader.get_path_to_datafile("_repeat_ops.so")
_repeat_ops = load_library.load_op_library(
  resource_loader.get_path_to_datafile("_repeat_ops.so"))


def repeat(input, repeats, axis=None, name=None):
  """Repeat elements of an array.

  Args:
    input: A Tensor.
    repeats: A 0 or 1-D `int` Tensor. The number of repetitions for each
      element, broadcast to fit the shape of the given axis.
    axis: An int. The axis along which to repeat values. If not given, the
      input is flattened and repeated along axis 0.
    name: name of the op.

  Returns:
    A Tensor which has the same shape as input, except along the given axis.
  """
  with ops.name_scope(name, "repeat"):
    if axis is None:
        input = array_ops.reshape(input, (-1,))
        axis = 0
    return _repeat_ops.repeat(
        input=input,
        repeats=repeats,
        axis=axis)


# not much performance benefit here (if any)
binary_repeat = _repeat_ops.binary_repeat
