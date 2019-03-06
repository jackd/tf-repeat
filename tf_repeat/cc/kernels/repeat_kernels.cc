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

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

class BinaryRepeatOp : public OpKernel {
 public:
  explicit BinaryRepeatOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {

  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& repeats_tensor = ctx->input(0);

    OP_REQUIRES(
        ctx,
        TensorShapeUtils::IsVector(repeats_tensor.shape()),
        errors::InvalidArgument("repeats must be 1-D, but got: ",
                                repeats_tensor.shape().DebugString()));
    auto repeats_flat = repeats_tensor.flat<int32>();

    Eigen::Tensor<int32, 0, Eigen::RowMajor> output_size = repeats_flat.sum();
    TensorShape output_shape{output_size()};

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
      ctx, ctx->allocate_output(0, output_shape, &output_tensor));

    auto output_flat = output_tensor->flat<bool>();

    bool value = false;
    auto repeats_len = repeats_flat.dimension(0);
    bool * output_arr = &output_flat(0);
    for (int64 i = 0; i < repeats_len; i++) {
      auto end = output_arr + repeats_flat(i);
      std::fill(output_arr, end, value);
      value = !value;
      output_arr = end;
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("BinaryRepeat").Device(DEVICE_CPU), BinaryRepeatOp);

template <typename T>
class RepeatOp : public OpKernel {
 public:
  explicit RepeatOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), axis_(0) {
    ctx->GetAttr("axis", &axis_);
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input_tensor = ctx->input(0);

    const Tensor& repeats_tensor = ctx->input(1);

    // repeats is either 0-D or 1-D
    OP_REQUIRES(
        ctx,
        TensorShapeUtils::IsScalar(repeats_tensor.shape()) ||
            TensorShapeUtils::IsVector(repeats_tensor.shape()),
        errors::InvalidArgument("repeats must be a 0-D or 1-D, but got: ",
                                repeats_tensor.shape().DebugString()));
    auto repeats_flat = repeats_tensor.flat<int32>();

    Tensor* output_tensor = nullptr;
    TensorShape output_shape(input_tensor.shape());
    if (TensorShapeUtils::IsScalar(input_tensor.shape())) {
      // If Scalar, then treat as [1]
      output_shape.AddDim(1);
    }
    OP_REQUIRES(ctx, (axis_ < output_shape.dims()),
                errors::InvalidArgument(
                    "axis must be < ", output_shape.dims(), ", got ", axis_));

    OP_REQUIRES(
        ctx,
        (repeats_flat.size() == 1 ||
          repeats_flat.size() == output_shape.dim_size(axis_)),
        errors::InvalidArgument(
            "repeats must have the same size as input, or 1, but got input ",
            output_shape.dim_size(axis_), ", repeats ", repeats_flat.size()));

    // reshape input so that axis is in the middle
    std::vector<int64> sizes{1, output_shape.dim_size(axis_), 1};
    if (axis_ > 0) {
      for (int64 i = 0; i < axis_; i++) {
        sizes[0] *= output_shape.dim_size(i);
      }
    }
    if (axis_ + 1 < output_shape.dims()) {
      for (int64 i = axis_ + 1; i < output_shape.dims(); i++) {
        sizes[2] *= output_shape.dim_size(i);
      }
    }
    auto input = input_tensor.shaped<T, 3>(sizes);

    if (repeats_flat.size() == 1) {
      output_shape.set_dim(
          axis_, input_tensor.shape().dim_size(axis_) * repeats_flat(0));
    } else {
      Eigen::Tensor<int32, 0, Eigen::RowMajor> output_size =
          repeats_flat.sum();
      output_shape.set_dim(axis_, output_size());
    }
    sizes[1] = output_shape.dim_size(axis_);

    OP_REQUIRES_OK(ctx,
                    ctx->allocate_output(0, output_shape, &output_tensor));
    auto output = output_tensor->shaped<T, 3>(sizes);

    int offset = 0;
    auto input_len = input.dimension(1);
    for (int64 i = 0; i < input_len; i++) {
      int64 repeats_value =
          repeats_flat.size() == 1 ? repeats_flat(0) : repeats_flat(i);
      int end = offset + repeats_value;
      for (int64 r = offset; r < end; r++) {
        output.chip(r, 1) = input.chip(i, 1);
      }
      offset += repeats_value;
    }
  }

 private:
  int axis_;
};

#define REGISTER_KERNEL(type)                                          \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("Repeat").Device(DEVICE_CPU).TypeConstraint<type>("T"),     \
      RepeatOp<type>);

TF_CALL_ALL_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL

}  // namespace tensorflow
