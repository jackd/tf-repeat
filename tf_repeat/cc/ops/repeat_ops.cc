/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


REGISTER_OP("Repeat")
    .Input("input: T")
    .Input("repeats: int32")
    .Output("output: T")
    .Attr("axis: int")
    .Attr("T: type")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      return Status::OK();
    })
    .Doc(R"doc(
Repeat elements of an array
input: A Tensor.
repeats: An 1-D `int` Tensor. The number of repetitions for each element.
  repeats is broadcasted to fit the shape of the given axis
axis: An int. The axis along which to repeat values. By default, use the
  flattened input array, and return a flat output array.
output: A Tensor which has the same shape as a, except along the given axis.
)doc");


REGISTER_OP("BinaryRepeat")
    .Input("repeats: int32")
    .Output("output: bool")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      return Status::OK();
    })
    .Doc(R"doc(
Repeat alternating 0s and 1s.
repeats: An 1-D `int` Tensor. The number of repetitions for each binary element.
output: A 1D bool Tensor.
)doc");
