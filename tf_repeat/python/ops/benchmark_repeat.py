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
"""Benchmark for various implementations of repeat ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import numpy as np

import tensorflow as tf
from repeat_ops import repeat

flags.DEFINE_integer("seed", default=132, help="numpy seed value")
flags.DEFINE_integer(
    "max_repeats", default=200, help="maximum entry in repeats")
flags.DEFINE_integer(
    "size", default=1000, help="size of inputs/repeats")
flags.DEFINE_integer(
    "burn_iters", default=20, help="number of iterations to burn-in")
flags.DEFINE_integer(
    "min_iters", default=100, help="minimum number of iterations to benchmark")


FLAGS = flags.FLAGS


def repeat_while(values, repeats):
  """Repeat using `tf.while_loop` with `Tensor` concatenation."""
  values = tf.convert_to_tensor(values)
  repeats = tf.convert_to_tensor(repeats)

  n_repeats = tf.size(repeats)
  _, accumulator = tf.while_loop(
      lambda i, _: tf.less(i, n_repeats),
      lambda i, acc:
        (i+1, tf.concat((acc, tf.fill([repeats[i]], values[i])), axis=0)),
      loop_vars=(
          tf.zeros_like(n_repeats),
          tf.zeros((0,), dtype=values.dtype)),
      back_prop=False,
      shape_invariants=(tf.TensorShape(()), tf.TensorShape((None,)))
  )

  return accumulator


def repeat_foldl(values, repeats):
  """Repeat using `tf.foldl` with `TensorArray` concatenation."""
  values = tf.convert_to_tensor(values)
  repeats = tf.convert_to_tensor(repeats)
  assert(repeats.shape.ndims == 1)
  assert(values.shape.ndims == 1)

  size = values.shape[0]
  indices = tf.range(size)
  element_shape = (None,) + tuple(values.shape[1:])
  acc0 = tf.TensorArray(
      dtype=values.dtype, size=size, dynamic_size=False,
      element_shape=element_shape, infer_shape=False)

  def fold_fn(acc, args):
    index, repeat, value = args
    value = tf.fill((repeat,), value)
    acc = acc.write(index, value)
    return acc

  acc = tf.foldl(fold_fn, (indices, repeats, values), initializer=acc0)
  return acc.concat()


def repeat_py_function(values, repeats):
  """Repeat using `tf.py_function` and `np.repeat`."""
  def fn(values, repeats):
    return np.repeat(values.numpy(), repeats.numpy())

  out = tf.py_function(fn, (values, repeats), values.dtype)
  out.set_shape((None,))
  return out


def run_benchmark(f, seed, max_repeats, size, burn_iters, min_iters):
  np.random.seed(seed)
  repeats = np.random.randint(max_repeats, size=size).astype(np.int32)
  values = np.random.randint(100, size=size).astype(np.int32)
  graph = tf.Graph()
  name = f.__name__
  with graph.as_default():
    with tf.device("/cpu:0"):
      out = f(values, repeats)

  with tf.Session(graph=graph) as sess:
    tf.logging.info("Benchmarking %s" % name)
    result = tf.test.Benchmark().run_op_benchmark(
        sess, out, burn_iters=burn_iters, min_iters=100, name=name)
    result = {k: v for k, v in result.items()}
    result['time_per_iter'] = result['wall_time'] / result['iters']
    return result


def main(argv):
  results = []
  tf.logging.set_verbosity(tf.logging.INFO)
  for fn in (
      repeat,
      repeat_py_function,
      repeat_foldl,
      repeat_while,
    ):
    results.append(run_benchmark(
        fn,
        seed=FLAGS.seed,
        max_repeats=FLAGS.max_repeats,
        size=FLAGS.size,
        burn_iters=FLAGS.burn_iters,
        min_iters=FLAGS.min_iters)
    )

  best = min(results, key=lambda r: r["time_per_iter"])
  best_time = best["time_per_iter"]
  longest_name = max(len(r["name"]) for r in results)
  for result in results:
    extra = "(x %2f)" % (result["time_per_iter"] / best_time)
    name = result["name"].ljust(longest_name)
    tf.logging.info("%s: %.3e %s" % (name, result["time_per_iter"], extra))

  tf.logging.info(
      "best: %s, time: %.3e" % (best["name"], best["time_per_iter"]))


if __name__ == "__main__":
  app.run(main)
