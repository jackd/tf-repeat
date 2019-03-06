CXX := g++
PYTHON_BIN_PATH = python

SRCS = $(wildcard tf_repeat/cc/kernels/*.cc) $(wildcard tf_repeat/cc/ops/*.cc)

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11
LDFLAGS = -shared ${TF_LFLAGS}

TARGET_LIB = tf_repeat/python/ops/_repeat_ops.so


.PHONY: op
op: $(TARGET_LIB)

$(TARGET_LIB): $(SRCS)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}

test: tf_repeat/python/ops/repeat_ops_test.py tf_repeat/python/ops/repeat_ops.py $(TARGET_LIB)
	$(PYTHON_BIN_PATH) tf_repeat/python/ops/repeat_ops_test.py


benchmark_repeat: tf_repeat/python/ops/benchmark_repeat.py tf_repeat/python/ops/repeat_ops.py $(TARGET_LIB)
	$(PYTHON_BIN_PATH) tf_repeat/python/ops/benchmark_repeat.py


benchmark_binary_repeat: tf_repeat/python/ops/benchmark_binary_repeat.py tf_repeat/python/ops/repeat_ops.py $(TARGET_LIB)
	$(PYTHON_BIN_PATH) tf_repeat/python/ops/benchmark_binary_repeat.py

pip_pkg: $(TARGET_LIB)
	./build_pip_pkg.sh make artifacts


.PHONY: clean
clean:
	rm -f $(TARGET_LIB)
