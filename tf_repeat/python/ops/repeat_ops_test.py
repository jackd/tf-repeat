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
"""Tests for repeat ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test
from repeat_ops import repeat


class RepeatTest(test.TestCase):

  def testRepeatScalar(self):
    with self.test_session():
      v_tf = repeat(constant_op.constant(3), 4)
      v_np = np.repeat(3, 4)
      self.assertAllEqual(v_tf.eval(), v_np)

  def testRepeatMatrix(self):
    with self.test_session():
      x = np.array([[1, 2], [3, 4]], dtype=np.int32)
      v_tf = repeat(constant_op.constant(x), 2)
      v_np = np.repeat(x, 2)
      self.assertAllEqual(v_tf.eval(), v_np)

  def testRepeatMatrixAxis0(self):
    with self.test_session():
      x = np.array([[1, 2], [3, 4]], dtype=np.int32)
      for axis in (0, 1):
        v_tf = repeat(
          constant_op.constant(x), constant_op.constant([1, 2]), axis=axis)
        v_np = np.repeat(x, [1, 2], axis=axis)
        self.assertAllEqual(v_tf.eval(), v_np)

  def testRepeatMatrixAxis1(self):
    with self.test_session():
      x = np.array([[1, 2], [3, 4]], dtype=np.int32)
      v_tf = repeat(constant_op.constant(x), constant_op.constant(3), axis=1)
      v_np = np.repeat(x, 3, axis=1)
      self.assertAllEqual(v_tf.eval(), v_np)

  def testRepeatMatrixRepeatArray(self):
    with self.test_session():
      x = np.array([[1, 2], [3, 4]], dtype=np.int32)
      v_tf = repeat(constant_op.constant(x), [1, 2, 3, 4])
      v_np = np.repeat(x, [1, 2, 3, 4])
      self.assertAllEqual(v_tf.eval(), v_np)

  def testRepeatDTypes(self):
    for dtype in [np.int8, np.int16, np.uint8, np.uint16, np.int32, np.int64]:
      with self.test_session():
        x = np.array([[1, 2], [3, 4]], dtype=dtype)
        v_tf = repeat(constant_op.constant(x), 2)
        v_np = np.repeat(x, 2)
        self.assertAllEqual(v_tf.eval(), v_np)


if __name__ == "__main__":
  test.main()
