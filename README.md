# [TensorFlow repeat op](https://github.com/jackd/tf-repeat)

Tensorflow implementation of [np.repeat](https://docs.scipy.org/doc/numpy/reference/generated/numpy.repeat.html), Adapted from [this pull request](https://github.com/tensorflow/tensorflow/pull/15224) using [custom-op](https://github.com/tensorflow/custom-op)

## Building

Clone and setup docker container.

```bash
git clone https://github.com/jackd/tf-repeat.git
cd tf-repeat
docker run -it -v ${PWD}:/working_dir -w /working_dir  tensorflow/tensorflow:custom-op
./configure
```

From inside the container, run `make`

```bash
make op
make test        # optional
make benchmark   # optional - see below
make pip_pkg
```

For further instructions (e.g. using `bazel`), see [custom-op](https://github.com/tensorflow/custom-op).

## Installing

```bash
cd artifacts
pip install tf_repeat-VERSION.whl
```

## Using

```python
import tensorflow as tf
from tf_repeat import repeat

repeat([2, 3, 4], [3, 2, 1])  # [2, 2, 2, 3, 3, 4]
```

See [tests](tf_repeat/python/ops/repeat_ops_test.py) for more examples.

## Benchmarks

```bash
>>> make benchmark
...
I0306 06:18:55.564097 22528204433216 repeat_ops_benchmark.py:135] repeat            : 1.352e-04 (x 1.000000)
I0306 06:18:55.564186 22528204433216 repeat_ops_benchmark.py:135] repeat_py_function: 7.740e-04 (x 5.725750)
I0306 06:18:55.564264 22528204433216 repeat_ops_benchmark.py:135] repeat_foldl      : 4.059e-02 (x 300.291887)
I0306 06:18:55.564332 22528204433216 repeat_ops_benchmark.py:135] repeat_while      : 8.552e-02 (x 632.597002)
I0306 06:18:55.564404 22528204433216 repeat_ops_benchmark.py:137] best: repeat, walltime: 1.352e-04
```

i.e. `repeat` is `~6x` faster than the implementation using `tf.py_function`, and `>300x` faster than methods using `tf.while` and `tf.foldl`.
