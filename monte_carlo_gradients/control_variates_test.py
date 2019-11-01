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

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from monte_carlo_gradients import control_variates
from monte_carlo_gradients import dist_utils
from monte_carlo_gradients import gradient_estimators
from monte_carlo_gradients import utils


class DeltaControlVariateTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters([(1.0, 0.5)])
  def testQuadraticFunction(self, effective_mean, effective_log_scale):
    data_dims = 20
    num_samples = 10**6

    mean = effective_mean * tf.ones(shape=(data_dims), dtype=tf.float32)
    log_scale = effective_log_scale * tf.ones(
        shape=(data_dims), dtype=tf.float32)

    dist = dist_utils.multi_normal(loc=mean, log_scale=log_scale)
    dist_samples = dist.sample(num_samples)
    function = lambda x: tf.reduce_sum(x**2)

    cv, expected_cv, _, _ = control_variates.control_delta_method(
        dist, dist_samples, function)
    avg_cv = tf.reduce_mean(cv)
    expected_cv_value = tf.reduce_sum(dist_samples** 2) / num_samples

    with self.test_session() as sess:
      init_op = tf.global_variables_initializer()
      sess.run(init_op)
      # This should be an analytical computation, the result needs to be
      # accurate.
      self.assertAllClose(
          sess.run(avg_cv), sess.run(expected_cv_value), rtol=1e-1, atol=1e-3)

      self.assertAllClose(
          sess.run(expected_cv),
          sess.run(expected_cv_value),
          atol=1e-1)

  @parameterized.parameters([(1.0, 1.0)])
  def testPolinomialFunction(self, effective_mean, effective_log_scale):
    data_dims = 10
    num_samples = 10**3

    mean = effective_mean * tf.ones(shape=(data_dims), dtype=tf.float32)
    log_scale = effective_log_scale * tf.ones(
        shape=(data_dims), dtype=tf.float32)

    dist = dist_utils.multi_normal(loc=mean, log_scale=log_scale)
    dist_samples = dist.sample(num_samples)
    function = lambda x: tf.reduce_sum(x**5)

    cv, expected_cv, _, _ = control_variates.control_delta_method(
        dist, dist_samples, function)
    avg_cv = tf.reduce_mean(cv)

    with self.test_session() as sess:
      init_op = tf.global_variables_initializer()
      sess.run(init_op)
      # Check that the average value of the control variate is close to the
      # expected value.
      self.assertAllClose(
          sess.run(avg_cv), sess.run(expected_cv), rtol=1e-1, atol=1e-3)

  @parameterized.parameters([(1.0, 1.0)])
  def testNonPolynomialFunction(self, effective_mean, effective_log_scale):
    data_dims = 10
    num_samples = 10**3

    mean = effective_mean * tf.ones(shape=(data_dims), dtype=tf.float32)
    log_scale = effective_log_scale * tf.ones(
        shape=(data_dims), dtype=tf.float32)

    dist = dist_utils.multi_normal(loc=mean, log_scale=log_scale)
    dist_samples = dist.sample(num_samples)
    function = lambda x: tf.reduce_sum(tf.log(x**2))

    cv, expected_cv, _, _ = control_variates.control_delta_method(
        dist, dist_samples, function)
    avg_cv = tf.reduce_mean(cv)

    self.assertTrue(tf.gradients(expected_cv, mean))
    self.assertTrue(tf.gradients(expected_cv, log_scale))

    with self.test_session() as sess:
      init_op = tf.global_variables_initializer()
      sess.run(init_op)
      # Check that the average value of the control variate is close to the
      # expected value.
      self.assertAllClose(
          sess.run(avg_cv), sess.run(expected_cv), rtol=1e-1, atol=1e-3)

  def testNonPolynomialFunctionWithGradients(self):
    data_dims = 1
    num_samples = 10**3
    effective_mean = 1.
    effective_log_scale = 1.

    mean = effective_mean * tf.ones(shape=(data_dims), dtype=tf.float32)
    log_scale = effective_log_scale * tf.ones(
        shape=(data_dims), dtype=tf.float32)

    dist = dist_utils.multi_normal(loc=mean, log_scale=log_scale)
    dist_samples = dist.sample(num_samples)
    function = lambda x: tf.reduce_sum(tf.log(x**2))

    (cv, expected_cv,
     surrogate_cv, jacobians) = control_variates.control_delta_method(
         dist, dist_samples, function,
         grad_loss_fn=utils.grad_loss_fn_with_jacobians(
             gradient_estimators.pathwise_loss))

    surrogate_cv = tf.reduce_mean(surrogate_cv)
    mean_cv_grads = tf.gradients(surrogate_cv, mean)[0]
    mean_expected_cv_grads = tf.gradients(expected_cv, mean)[0]

    log_scale_cv_grads = tf.gradients(surrogate_cv, log_scale)[0]
    log_scale_expected_cv_grads = tf.gradients(expected_cv, log_scale)[0]

    # Second order expansion is log(\mu**2) + 1/2 * \sigma**2 (-2 / \mu**2)
    expected_cv_val = - np.exp(1.) ** 2

    # The gradient is 2 / mu + \sigma ** 2 * 2
    expected_cv_mean_grad = 2 + 2 * np.exp(1.) ** 2

    mean_jacobians = jacobians[mean]
    log_scale_jacobians = jacobians[log_scale]

    with self.test_session() as sess:
      init_op = tf.global_variables_initializer()
      sess.run(init_op)

      self.assertAllClose(
          sess.run(tf.reduce_mean(cv)),
          sess.run(expected_cv), rtol=1e-1, atol=1e-3)

      self.assertAllClose(
          sess.run(expected_cv), expected_cv_val, rtol=1e-1, atol=1e-3)

      self.assertAllClose(
          sess.run(tf.reduce_mean(cv)), expected_cv_val, rtol=1e-1, atol=1e-3)

      self.assertAllClose(
          sess.run(mean_expected_cv_grads[0]), expected_cv_mean_grad,
          rtol=1e-1, atol=1e-3)

      self.assertAllClose(
          sess.run(mean_cv_grads),
          sess.run(mean_expected_cv_grads), rtol=1e-1, atol=1e-3)

      self.assertAllClose(
          sess.run(log_scale_cv_grads),
          sess.run(log_scale_expected_cv_grads), rtol=1e-1, atol=1e-3)

      self.assertAllClose(
          sess.run(tf.reduce_mean(mean_jacobians)),
          # Strip the leading dimension of 1.
          sess.run(mean_cv_grads[0]), rtol=1e-1, atol=1e-3)

      self.assertAllClose(
          sess.run(tf.reduce_mean(log_scale_jacobians)),
          # Strip the leading dimension of 1.
          sess.run(log_scale_cv_grads[0]), rtol=1e-1, atol=1e-3)


class SurrogateLossFromGradients(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters([
      (1.0, 1.0, gradient_estimators.score_function_loss, True),
      (1.0, 1.0, gradient_estimators.pathwise_loss, False),
      (1.0, 1.0, gradient_estimators.pathwise_loss, True)])
  def testQuadraticFunction(
      self, effective_mean, effective_log_scale, grad_loss_fn,
      estimate_cv_coeff):
    data_dims = 3
    num_samples = 10**3

    mean = effective_mean * tf.ones(shape=(data_dims), dtype=tf.float32)
    log_scale = effective_log_scale * tf.ones(
        shape=(data_dims), dtype=tf.float32)

    dist_vars = [mean, log_scale]
    dist = dist_utils.multi_normal(loc=mean, log_scale=log_scale)
    dist_samples = dist.sample(num_samples)

    model_loss_fn = lambda x: tf.reduce_sum(x**2, axis=1)
    control_variate_fn = control_variates.control_delta_method

    loss, jacobians = control_variates.control_variates_surrogate_loss(
        dist=dist,
        dist_samples=dist_samples,
        dist_vars=dist_vars,
        model_loss_fn=model_loss_fn,
        grad_loss_fn=utils.grad_loss_fn_with_jacobians(grad_loss_fn),
        estimate_cv_coeff=estimate_cv_coeff,
        control_variate_fn=control_variate_fn)

    loss.shape.assert_is_compatible_with([num_samples])
    loss = tf.reduce_mean(loss)

    expected_mean_grads = 2 * effective_mean * np.ones(
        data_dims, dtype=np.float32)
    expected_log_scale_grads = 2 * np.exp(2 * effective_log_scale) * np.ones(
        data_dims, dtype=np.float32)

    mean_jacobians = jacobians[mean]
    mean_jacobians.shape.assert_is_compatible_with([num_samples, data_dims])
    mean_grads_from_jacobian = tf.reduce_mean(mean_jacobians, axis=0)

    log_scale_jacobians = jacobians[log_scale]
    log_scale_jacobians.shape.assert_is_compatible_with(
        [num_samples, data_dims])
    log_scale_grads_from_jacobian = tf.reduce_mean(log_scale_jacobians, axis=0)

    mean_grads = tf.gradients(loss, mean)[0]
    mean_grads.shape.assert_is_compatible_with(data_dims)

    log_scale_grads = tf.gradients(loss, log_scale)[0]
    log_scale_grads.shape.assert_is_compatible_with(data_dims)

    with self.test_session() as sess:
      init_op = tf.initialize_all_variables()
      sess.run(init_op)

      self.assertAllClose(
          sess.run(mean_grads), expected_mean_grads, rtol=1e-1, atol=1e-3)
      self.assertAllClose(
          sess.run(log_scale_grads),
          expected_log_scale_grads, rtol=1e-1, atol=1e-3)

      self.assertAllClose(
          sess.run(mean_grads_from_jacobian), expected_mean_grads,
          rtol=1e-1, atol=1e-3)
      self.assertAllClose(
          sess.run(log_scale_grads_from_jacobian),
          expected_log_scale_grads, rtol=1e-1, atol=1e-3)

  @parameterized.parameters([
      (1.0, 1.0, gradient_estimators.score_function_loss),
      (1.0, 1.0, gradient_estimators.pathwise_loss)])
  def testQuadraticFunctionWithAnalyticalLoss(
      self, effective_mean, effective_log_scale, grad_loss_fn):
    data_dims = 3
    num_samples = 10**3

    mean = effective_mean * tf.ones(shape=(data_dims), dtype=tf.float32)
    log_scale = effective_log_scale * tf.ones(
        shape=(data_dims), dtype=tf.float32)

    dist_vars = [mean, log_scale]
    dist = dist_utils.multi_normal(loc=mean, log_scale=log_scale)
    dist_samples = dist.sample(num_samples)

    model_loss_fn = lambda x: tf.reduce_sum(x**2, axis=1)
    control_variate_fn = control_variates.control_delta_method

    loss, jacobians = control_variates.control_variates_surrogate_loss(
        dist=dist,
        dist_samples=dist_samples,
        dist_vars=dist_vars,
        model_loss_fn=model_loss_fn,
        grad_loss_fn=utils.grad_loss_fn_with_jacobians(grad_loss_fn),
        control_variate_fn=control_variate_fn)

    loss.shape.assert_is_compatible_with([num_samples])
    loss = tf.reduce_mean(loss)

    expected_mean_grads = 2 * effective_mean * np.ones(
        data_dims, dtype=np.float32)
    expected_log_scale_grads = 2 * np.exp(2 * effective_log_scale) * np.ones(
        data_dims, dtype=np.float32)

    mean_jacobians = jacobians[mean]
    mean_jacobians.shape.assert_is_compatible_with([num_samples, data_dims])
    mean_grads_from_jacobian = tf.reduce_mean(mean_jacobians, axis=0)

    log_scale_jacobians = jacobians[log_scale]
    log_scale_jacobians.shape.assert_is_compatible_with(
        [num_samples, data_dims])
    log_scale_grads_from_jacobian = tf.reduce_mean(log_scale_jacobians, axis=0)

    mean_grads = tf.gradients(loss, mean)[0]
    mean_grads.shape.assert_is_compatible_with(data_dims)

    log_scale_grads = tf.gradients(loss, log_scale)[0]
    log_scale_grads.shape.assert_is_compatible_with(data_dims)

    with self.test_session() as sess:
      init_op = tf.initialize_all_variables()
      sess.run(init_op)

      self.assertAllClose(
          sess.run(mean_grads), expected_mean_grads, rtol=1e-1, atol=1e-3)
      self.assertAllClose(
          sess.run(log_scale_grads),
          expected_log_scale_grads, rtol=1e-1, atol=1e-3)

      self.assertAllClose(
          sess.run(mean_grads_from_jacobian), expected_mean_grads,
          rtol=1e-1, atol=1e-3)
      self.assertAllClose(
          sess.run(log_scale_grads_from_jacobian),
          expected_log_scale_grads, rtol=1e-1, atol=1e-3)

  @parameterized.parameters([
      (1.0, 1.0, gradient_estimators.score_function_loss, 7 * 10 **3),
      (1.0, 1.0, gradient_estimators.measure_valued_loss, 10 **3),
      (1.0, 1.0, gradient_estimators.pathwise_loss, 10 **3)])
  def testNonPolynomialFunctionConsistency(
      self, effective_mean, effective_log_scale, grad_loss_fn, num_samples):
    """Check that the gradients are consistent between estimators."""
    data_dims = 3

    mean = effective_mean * tf.ones(shape=(data_dims), dtype=tf.float32)
    log_scale = effective_log_scale * tf.ones(
        shape=(data_dims), dtype=tf.float32)

    dist_vars = [mean, log_scale]
    dist = dist_utils.multi_normal(loc=mean, log_scale=log_scale)
    dist_samples = dist.sample(num_samples)

    model_loss_fn = lambda x: tf.log(tf.reduce_sum(x**2, axis=1))
    control_variate_fn = control_variates.control_delta_method

    grad_loss_fn = utils.grad_loss_fn_with_jacobians(grad_loss_fn)
    loss, jacobians = control_variates.control_variates_surrogate_loss(
        dist=dist,
        dist_samples=dist_samples,
        dist_vars=dist_vars,
        model_loss_fn=model_loss_fn,
        grad_loss_fn=grad_loss_fn,
        control_variate_fn=control_variate_fn)

    loss.shape.assert_is_compatible_with([num_samples])
    loss = tf.reduce_mean(loss)

    mean_jacobians = jacobians[mean]
    mean_jacobians.shape.assert_is_compatible_with([num_samples, data_dims])
    mean_grads_from_jacobian = tf.reduce_mean(mean_jacobians, axis=0)

    log_scale_jacobians = jacobians[log_scale]
    log_scale_jacobians.shape.assert_is_compatible_with(
        [num_samples, data_dims])
    log_scale_grads_from_jacobian = tf.reduce_mean(log_scale_jacobians, axis=0)

    mean_grads = tf.gradients(loss, mean)[0]
    mean_grads.shape.assert_is_compatible_with(data_dims)

    log_scale_grads = tf.gradients(loss, log_scale)[0]
    log_scale_grads.shape.assert_is_compatible_with(data_dims)

    no_cv_loss, _ = grad_loss_fn(model_loss_fn, dist_samples, dist)
    no_cv_loss.shape.assert_is_compatible_with([num_samples])
    no_cv_loss = tf.reduce_mean(no_cv_loss)

    no_cv_mean_grads = tf.gradients(no_cv_loss, mean)[0]
    no_cv_mean_grads.shape.assert_is_compatible_with(data_dims)

    no_cv_log_scale_grads = tf.gradients(no_cv_loss, log_scale)[0]
    no_cv_log_scale_grads.shape.assert_is_compatible_with(data_dims)

    with self.test_session() as sess:
      init_op = tf.initialize_all_variables()
      sess.run(init_op)

      (mean_grads_from_jacobian_np, mean_grads_np,
       log_scale_grads_from_jacobian_np, log_scale_grads_np,
       no_cv_mean_grads_np, no_cv_log_scale_grads_np) = sess.run(
           [mean_grads_from_jacobian, mean_grads,
            log_scale_grads_from_jacobian, log_scale_grads,
            no_cv_mean_grads, no_cv_log_scale_grads])

      self.assertAllClose(
          mean_grads_from_jacobian_np, mean_grads_np, rtol=1e-1, atol=1e-3)

      self.assertAllClose(
          log_scale_grads_from_jacobian_np, log_scale_grads_np,
          rtol=1e-1, atol=1e-3)

      self.assertAllClose(
          mean_grads_np, no_cv_mean_grads_np, rtol=1e-1, atol=1e-1)

      self.assertAllClose(
          log_scale_grads_np, no_cv_log_scale_grads_np, rtol=1e-1, atol=1e-1)


if __name__ == '__main__':
  tf.test.main()
