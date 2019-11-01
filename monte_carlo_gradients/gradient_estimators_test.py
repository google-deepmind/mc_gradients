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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from monte_carlo_gradients import dist_utils
from monte_carlo_gradients import gradient_estimators


def _cross_prod(items1, items2):
  prod = itertools.product(items1, items2)
  return [i1 + (i2,) for i1, i2 in prod]


class ReinforceTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters([0.1, 0.5, 0.9])
  def testConstantFunction(self, constant):
    data_dims = 3
    num_samples = 10**6

    effective_mean = 1.5
    mean = effective_mean * tf.ones(shape=(data_dims), dtype=tf.float32)

    effective_log_scale = 0.0
    log_scale = effective_log_scale * tf.ones(
        shape=(data_dims), dtype=tf.float32)

    dist = dist_utils.multi_normal(loc=mean, log_scale=log_scale)
    dist_samples = dist.sample(num_samples)
    dist_samples.shape.assert_is_compatible_with([num_samples, data_dims])

    function = lambda x: tf.ones_like(x[:, 0])
    loss = gradient_estimators.score_function_loss(
        function, dist_samples, dist)

    # Average over the number of samples.
    loss.shape.assert_is_compatible_with([num_samples])
    loss = tf.reduce_mean(loss)

    mean_grads = tf.gradients(loss, mean)[0]
    mean_grads.shape.assert_is_compatible_with(data_dims)
    expected_mean_grads = np.zeros(data_dims, dtype=np.float32)

    log_scale_grads = tf.gradients(loss, log_scale)[0]
    expected_log_scale_grads = np.zeros(data_dims, dtype=np.float32)

    with self.test_session() as sess:
      init_op = tf.global_variables_initializer()
      sess.run(init_op)

      self.assertAllClose(
          sess.run(mean_grads), expected_mean_grads, rtol=1e-1, atol=5e-3)
      self.assertAllClose(
          sess.run(log_scale_grads),
          expected_log_scale_grads,
          rtol=1e-1,
          atol=5e-3)

  @parameterized.parameters([(0.5, -1.), (0.7, 0.0), (0.8, 0.1)])
  def testLinearFunction(self, effective_mean, effective_log_scale):
    data_dims = 3
    num_samples = 10**6

    mean = effective_mean * tf.ones(shape=(data_dims), dtype=tf.float32)
    log_scale = effective_log_scale * tf.ones(
        shape=(data_dims), dtype=tf.float32)

    dist = dist_utils.multi_normal(loc=mean, log_scale=log_scale)
    dist_samples = dist.sample(num_samples)
    dist_samples.shape.assert_is_compatible_with([num_samples, data_dims])

    function = lambda x: tf.reduce_sum(x, axis=1)
    loss = gradient_estimators.score_function_loss(
        dist=dist, dist_samples=dist_samples, function=function)
    loss.shape.assert_is_compatible_with([num_samples])
    loss = tf.reduce_mean(loss)

    mean_grads = tf.gradients(loss, mean)[0]
    mean_grads.shape.assert_is_compatible_with(data_dims)
    expected_mean_grads = np.ones(data_dims, dtype=np.float32)

    log_scale_grads = tf.gradients(loss, log_scale)[0]
    expected_log_scale_grads = np.zeros(data_dims, dtype=np.float32)

    with self.test_session() as sess:
      init_op = tf.global_variables_initializer()
      sess.run(init_op)

      self.assertAllClose(
          sess.run(mean_grads), expected_mean_grads, rtol=1e-1, atol=1e-2)
      self.assertAllClose(
          sess.run(log_scale_grads),
          expected_log_scale_grads,
          rtol=1e-1,
          atol=1e-2)

  @parameterized.parameters([(1.0, 1.0)])
  def testQuadraticFunction(self, effective_mean, effective_log_scale):
    data_dims = 3
    num_samples = 10**6

    mean = effective_mean * tf.ones(shape=(data_dims), dtype=tf.float32)
    log_scale = effective_log_scale * tf.ones(
        shape=(data_dims), dtype=tf.float32)

    dist = dist_utils.multi_normal(loc=mean, log_scale=log_scale)
    dist_samples = dist.sample(num_samples)
    function = lambda x: tf.reduce_sum(x**2, axis=1)

    loss = gradient_estimators.score_function_loss(
        dist=dist,
        dist_samples=dist_samples,
        function=function)
    loss.shape.assert_is_compatible_with([num_samples])
    loss = tf.reduce_mean(loss)

    mean_grads = tf.gradients(loss, mean)[0]
    mean_grads.shape.assert_is_compatible_with(data_dims)
    expected_mean_grads = 2 * effective_mean * np.ones(
        data_dims, dtype=np.float32)

    log_scale_grads = tf.gradients(loss, log_scale)[0]
    log_scale_grads.shape.assert_is_compatible_with(data_dims)
    expected_log_scale_grads = 2 * np.exp(2 * effective_log_scale) * np.ones(
        data_dims, dtype=np.float32)

    with self.test_session() as sess:
      init_op = tf.initialize_all_variables()
      sess.run(init_op)

      self.assertAllClose(
          sess.run(mean_grads), expected_mean_grads, rtol=1e-1, atol=1e-3)
      self.assertAllClose(
          sess.run(log_scale_grads),
          expected_log_scale_grads, rtol=1e-1, atol=1e-3)


class ReparametrizationTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters([1.2, 10.0, 5.])
  def testConstantFunction(self, constant):
    data_dims = 3
    num_samples = 10**6

    effective_mean = 1.5
    mean = effective_mean * tf.ones(shape=(data_dims), dtype=tf.float32)

    effective_log_scale = 0.0
    log_scale = effective_log_scale * tf.ones(
        shape=(data_dims), dtype=tf.float32)

    dist = dist_utils.multi_normal(loc=mean, log_scale=log_scale)
    dist_samples = dist.sample(num_samples)
    dist_samples.shape.assert_is_compatible_with([num_samples, data_dims])

    function = lambda x: tf.ones_like(x[:, 0])
    loss = gradient_estimators.pathwise_loss(
        function, dist_samples, dist)

    loss.shape.assert_is_compatible_with([num_samples])
    loss = tf.reduce_mean(loss)

    mean_grads = tf.gradients(loss, mean)[0]
    self.assertFalse(mean_grads)

    log_scale_grads = tf.gradients(loss, log_scale)[0]
    self.assertFalse(log_scale_grads)

  @parameterized.parameters([(0.5, -1.), (1.0, 0.0), (0.8, 0.1)])
  def testLinearFunction(self, effective_mean, effective_log_scale):
    data_dims = 3
    num_samples = 10**6

    mean = effective_mean * tf.ones(shape=(data_dims), dtype=tf.float32)
    log_scale = effective_log_scale * tf.ones(
        shape=(data_dims), dtype=tf.float32)

    dist = dist_utils.multi_normal(loc=mean, log_scale=log_scale)
    dist_samples = dist.sample(num_samples)
    dist_samples.shape.assert_is_compatible_with([num_samples, data_dims])

    function = lambda x: tf.reduce_sum(x, axis=1)
    loss = gradient_estimators.pathwise_loss(
        function, dist_samples, dist)
    loss.shape.assert_is_compatible_with([num_samples])
    loss = tf.reduce_mean(loss)
    loss.shape.assert_is_compatible_with([])

    mean_grads = tf.gradients(loss, mean)[0]
    mean_grads.shape.assert_is_compatible_with(data_dims)
    expected_mean_grads = np.ones(data_dims, dtype=np.float32)

    log_scale_grads = tf.gradients(loss, log_scale)[0]
    expected_log_scale_grads = np.zeros(data_dims, dtype=np.float32)

    with self.test_session() as sess:
      init_op = tf.global_variables_initializer()
      sess.run(init_op)
      # This should be an analytical computation, the result needs to be
      # accurate.
      self.assertAllClose(
          sess.run(mean_grads), expected_mean_grads, rtol=1e-1, atol=1e-3)

      self.assertAllClose(
          sess.run(log_scale_grads),
          expected_log_scale_grads,
          atol=1e-2)

  @parameterized.parameters([(1.0, 1.0)])
  def testQuadraticFunction(self, effective_mean, effective_log_scale):
    data_dims = 1
    num_samples = 10**6

    mean = effective_mean * tf.ones(shape=(data_dims), dtype=tf.float32)
    log_scale = effective_log_scale * tf.ones(
        shape=(data_dims), dtype=tf.float32)

    dist = dist_utils.multi_normal(loc=mean, log_scale=log_scale)
    dist_samples = dist.sample(num_samples)
    function = lambda x: tf.reduce_sum(x**2, axis=1)

    loss = gradient_estimators.pathwise_loss(
        function, dist_samples, dist)
    loss.shape.assert_is_compatible_with([num_samples])
    loss = tf.reduce_mean(loss)
    loss.shape.assert_is_compatible_with([])

    mean_grads = tf.gradients(loss, mean)[0]
    mean_grads.shape.assert_is_compatible_with(data_dims)
    expected_mean_grads = 2 * effective_mean * np.ones(
        data_dims, dtype=np.float32)

    log_scale_grads = tf.gradients(loss, log_scale)[0]
    log_scale_grads.shape.assert_is_compatible_with(data_dims)
    expected_log_scale_grads = 2 * np.exp(2 * effective_log_scale) * np.ones(
        data_dims, dtype=np.float32)

    with self.test_session() as sess:
      init_op = tf.initialize_all_variables()
      sess.run(init_op)

      self.assertAllClose(
          sess.run(mean_grads), expected_mean_grads, rtol=1e-1, atol=1e-3)
      self.assertAllClose(
          sess.run(log_scale_grads),
          expected_log_scale_grads, rtol=1e-1, atol=1e-3)


class MeasureValuedDerivativesTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(_cross_prod(
      [([1.0, 2.0, 3.], [-1., 0.3, -2.], [1., 1., 1.]),
       ([1.0, 2.0, 3.], [-1., 0.3, -2.], [4., 2., 3.]),
       ([1.0, 2.0, 3.], [0.1, 0.2, 0.1], [10., 5., 1.])
      ], [True, False]))
  def testWeightedLinear(
      self, effective_mean, effective_log_scale, weights, coupling):
    num_samples = 10**5

    effective_mean = np.array(effective_mean)
    effective_log_scale = np.array(effective_log_scale)
    weights = np.array(weights)

    data_dims = len(effective_mean)

    mean = tf.constant(effective_mean, dtype=tf.float32)
    log_scale = tf.constant(effective_log_scale, dtype=tf.float32)

    dist = dist_utils.multi_normal(loc=mean, log_scale=log_scale)
    dist_samples = dist.sample(num_samples)

    function = lambda x: (tf.reduce_sum(x* weights, axis=1))
    loss, _ = gradient_estimators.measure_valued_loss(
        function, dist_samples, dist, coupling=coupling)

    loss.shape.assert_is_compatible_with([num_samples])
    loss = tf.reduce_mean(loss)

    mean_grads = tf.gradients(loss, mean)[0]
    mean_grads.shape.assert_is_compatible_with(data_dims)

    log_scale_grads = tf.gradients(loss, log_scale)[0]
    log_scale_grads.shape.assert_is_compatible_with(data_dims)

    expected_mean_grads = weights
    expected_log_scale_grads = np.zeros(data_dims, dtype=np.float32)

    with self.test_session() as sess:
      init_op = tf.initialize_all_variables()
      sess.run(init_op)

      mean_grads_np, log_scale_grads_np = sess.run(
          [mean_grads, log_scale_grads])
      self.assertAllClose(
          expected_mean_grads, mean_grads_np, rtol=1e-1, atol=1e-3)
      self.assertAllClose(
          expected_log_scale_grads, log_scale_grads_np, rtol=1, atol=1e-3)

  @parameterized.parameters(_cross_prod(
      [([1.0, 2.0, 3.], [-1., 0.3, -2.], [1., 1., 1.]),
       ([1.0, 2.0, 3.], [-1., 0.3, -2.], [4., 2., 3.]),
       ([1.0, 2.0, 3.], [0.1, 0.2, 0.1], [10., 5., 1.])
      ], [True, False]))
  def testWeightedQuadratic(
      self, effective_mean, effective_log_scale, weights, coupling):
    num_samples = 5 * 10**5

    effective_mean = np.array(effective_mean)
    effective_log_scale = np.array(effective_log_scale)
    weights = np.array(weights)

    data_dims = len(effective_mean)

    mean = tf.constant(effective_mean, dtype=tf.float32)
    log_scale = tf.constant(effective_log_scale, dtype=tf.float32)

    dist = dist_utils.multi_normal(loc=mean, log_scale=log_scale)
    dist_samples = dist.sample(num_samples)

    function = lambda x: (tf.reduce_sum(x* weights, axis=1)) ** 2
    loss, _ = gradient_estimators.measure_valued_loss(
        function, dist_samples, dist, coupling=coupling)

    loss.shape.assert_is_compatible_with([num_samples])
    loss = tf.reduce_mean(loss)

    mean_grads = tf.gradients(loss, mean)[0]
    mean_grads.shape.assert_is_compatible_with(data_dims)

    log_scale_grads = tf.gradients(loss, log_scale)[0]
    log_scale_grads.shape.assert_is_compatible_with(data_dims)

    expected_mean_grads = 2 * weights * np.sum(weights * effective_mean)
    effective_scale = np.exp(effective_log_scale)
    expected_scale_grads = 2 * weights ** 2 * effective_scale
    expected_log_scale_grads = expected_scale_grads * effective_scale

    with self.test_session() as sess:
      init_op = tf.initialize_all_variables()
      sess.run(init_op)

      mean_grads_np, log_scale_grads_np = sess.run(
          [mean_grads, log_scale_grads])
      self.assertAllClose(
          expected_mean_grads, mean_grads_np, rtol=1e-1, atol=1e-1)
      self.assertAllClose(
          expected_log_scale_grads, log_scale_grads_np, rtol=1e-1, atol=1e-1)

  @parameterized.parameters(_cross_prod(
      [([1.0], [1.0], lambda x: (tf.reduce_sum(tf.cos(x), axis=1))),
       # Need to ensure that the mean is not too close to 0.
       ([10.0], [0.0], lambda x: (tf.reduce_sum(tf.log(x), axis=1))),
       ([1.0, 2.0], [1.0, -2],
        lambda x: (tf.reduce_sum(tf.cos(2 * x), axis=1))),
       ([1.0, 2.0], [1.0, -2],
        lambda x: (tf.cos(tf.reduce_sum(2 * x, axis=1)))),
      ], [True, False]))
  def testNonPolynomialFunctionConsistencyWithReparam(
      self, effective_mean, effective_log_scale, function, coupling):
    num_samples = 10**5

    effective_mean = np.array(effective_mean)
    effective_log_scale = np.array(effective_log_scale)
    data_dims = len(effective_mean)

    mean = tf.constant(effective_mean, dtype=tf.float32)
    log_scale = tf.constant(effective_log_scale, dtype=tf.float32)

    dist = dist_utils.multi_normal(loc=mean, log_scale=log_scale)
    dist_samples = dist.sample(num_samples)

    loss, _ = gradient_estimators.measure_valued_loss(
        function, dist_samples, dist, coupling=coupling)

    loss.shape.assert_is_compatible_with([num_samples])
    loss = tf.reduce_mean(loss)

    mean_grads = tf.gradients(loss, mean)[0]
    mean_grads.shape.assert_is_compatible_with(data_dims)

    log_scale_grads = tf.gradients(loss, log_scale)[0]
    log_scale_grads.shape.assert_is_compatible_with(data_dims)

    reparam_loss = gradient_estimators.pathwise_loss(
        function, dist_samples, dist)

    reparam_loss.shape.assert_is_compatible_with([num_samples])
    reparam_loss = tf.reduce_mean(reparam_loss)

    reparam_mean_grads = tf.gradients(reparam_loss, mean)[0]
    reparam_log_scale_grads = tf.gradients(reparam_loss, log_scale)[0]

    with self.test_session() as sess:
      init_op = tf.initialize_all_variables()
      sess.run(init_op)

      (mean_grads_np, log_scale_grads_np,
       reparam_mean_grads_np, reparam_log_scale_grads_np) = sess.run(
           [mean_grads, log_scale_grads,
            reparam_mean_grads, reparam_log_scale_grads])
      self.assertAllClose(
          reparam_mean_grads_np, mean_grads_np, rtol=5e-1, atol=1e-1)
      self.assertAllClose(
          reparam_log_scale_grads_np, log_scale_grads_np, rtol=5e-1, atol=1e-1)


if __name__ == '__main__':
  tf.test.main()
