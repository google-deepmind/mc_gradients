# Copyright 2019 Deepmind Technologies Limited.
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
"""General utilities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

import tensorflow as tf
from tensorflow.python.ops.parallel_for import gradients as pfor_gradients  # pylint: disable=g-direct-tensorflow-import


def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if name is None:
    name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape


def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  if name is None:
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.get_variable_scope().name
    raise ValueError(
        "For the tensor `%s` in scope `%s`, the actual rank "
        "`%d` (shape = %s) is not equal to the expected rank `%s`" %
        (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def tile_second_to_last_dim(t):
  rank = t.shape.rank
  multiples = [1] * (rank - 1) + [tf.shape(t)[-1]] + [1]
  return tf.tile(tf.expand_dims(t, axis=-2), multiples)


def jacobians(ys, xs, parallel_iterations=None):
  """Compute the jacobians of `ys` with respect to `xs`.

  Args:
    ys: tf.Tensor.
    xs: tf.Tensor. The variables wrt to compute the Jacobian.
    parallel_iterations: The number of iterations to be done in paralel. Used to
        trade-off memory consumption for speed: if None, the Jacobian
        computation is done in parallel, but requires most memory.

  Returns:
    a tf.Tensor of Jacobians.
  """
  return pfor_gradients.jacobian(
      ys, xs, use_pfor=True, parallel_iterations=parallel_iterations)


def add_grads_to_jacobians(jacobians_dict, y, x_vars=None):
  if not x_vars:
    x_vars = jacobians_dict.keys()

  for var in x_vars:
    grads = tf.gradients(y, var)[0]
    if grads is not None:
      jacobians_dict[var] += grads

  return jacobians_dict


def hessians(ys, xs, no_backprop=False):
  if not no_backprop:
    return tf.squeeze(tf.hessians(ys, xs)[0], axis=[0, 2])
  grads = tf.gradients(ys, xs)[0][0]
  # Note: it is important to use parallel_iterations=None here, to avoid an
  # in graph while loop inside the jacobians computation, which itself is an
  # in graph while loop. This is more efficient since we do not have a large
  # number of parameters, but can use many samples to compute gradient estimator
  # variance.
  return tf.squeeze(
      jacobians(grads, xs, parallel_iterations=None), axis=1)


def dist_var_jacobians(loss, dist, parallel_iterations=None):
  return {p: jacobians(loss, p, parallel_iterations=parallel_iterations)
          for p in dist.dist_vars}


def grad_loss_fn_with_jacobians(
    gradient_estimator_fn, jacobian_parallel_iterations=None):
  """Wrap a gradient loss function to return a surrogate loss and jacobians."""
  def grad_fn(function, dist_samples, dist):
    output = gradient_estimator_fn(function, dist_samples, dist)
    if isinstance(output, tf.Tensor):
      loss = output
      jacobian_dict = dist_var_jacobians(
          loss, dist, parallel_iterations=jacobian_parallel_iterations)
    else:
      loss, jacobian_dict = output

    return loss, jacobian_dict

  return grad_fn
