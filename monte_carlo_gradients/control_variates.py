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
"""Control variate utils."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sonnet as snt
import tensorflow as tf

from monte_carlo_gradients import utils


def control_delta_method(dist, dist_samples, function, grad_loss_fn=None):
  """Computes the delta method control variate.

  For details, see: https://icml.cc/2012/papers/687.pdf

  Args:
    dist: A tfp.distributions.Distribution instance.
      The expected control variate computation assumes this distribution is
      from the Normal family.
    dist_samples: a tf.Tensor of samples from `dist`.
    function: A function for which to compute the second order approximation.
    grad_loss_fn: The gradient estimator function or None.
      Needs to return both a surrogate loss and a dictionary of jacobians.
      If None, only the first two elements in the return tuple are meaningful.

  Returns:
    A tuple containing four elements:
      * The control variate, a [B, num_samples] tensor. The
        number of samples is taken from `model_outputs.posterior_samples`.
      * The expected value of the control variate, a [B] tensor.
      * The surrogate CV used for gradient estimation (a `tf.Tensor`).
        This tensor is obtained by splitting the control variate into
        two parts: one for the stochastic gradient estimation (via
        `grad_loss_fn`) and one via analytical gradients. This is
        done via the chain rule. Note: the value of this loss will not
        be the same as the value of the control variate - but twice
        its value.
      * A dictionary containing the jacobians of the control variate according
          to `grad_fn`. If `grad_fn` is None, then a dictionary with 0 values.
  """
  mean_dist = dist.input_mean
  # Expand the mean distribution tensor.
  data_dim = utils.get_shape_list(dist_samples)[1]
  mean_dist = tf.ones((1, 1)) * tf.expand_dims(mean_dist, axis=0)

  def _cv(mean_dist, dist_samples, stop_grad_mean):
    """Computes the value of the control variate."""
    num_samples = utils.get_shape_list(dist_samples)[0]

    if stop_grad_mean:
      mean_dist = tf.stop_gradient(mean_dist)

    posterior_mean_f = function(mean_dist)
    # The loss has been summed over batch, and we only used one parameter
    # (the mean) to compute loss.
    posterior_mean_f_shape = utils.get_shape_list(posterior_mean_f)
    if posterior_mean_f_shape not in [[1], []]:
      raise ValueError('Invalid shape for posterior_mean_f {}!'.format(
          posterior_mean_f_shape))

    # Compute first order gradients.
    first_order_gradients = tf.gradients(posterior_mean_f, mean_dist)[0]
    first_order_gradients.shape.assert_is_compatible_with([1, data_dim])

    # Compute second order gradients.
    second_order_gradients = utils.hessians(
        posterior_mean_f, mean_dist, no_backprop=stop_grad_mean)
    second_order_gradients.shape.assert_is_compatible_with(
        [data_dim, data_dim])

    centered_dist_samples = dist_samples  - mean_dist
    centered_dist_samples.shape.assert_is_compatible_with(
        [num_samples, data_dim])

    first_order_term = tf.reduce_sum(
        centered_dist_samples * first_order_gradients, axis=1)
    first_order_term.shape.assert_is_compatible_with([num_samples])

    second_order_term = tf.matmul(
        centered_dist_samples, second_order_gradients)
    second_order_term.shape.assert_is_compatible_with(
        [num_samples, data_dim])

    second_order_term = tf.reduce_sum(
        second_order_term * centered_dist_samples, axis=1)
    second_order_term = 1./2 * second_order_term

    control_variate = posterior_mean_f + first_order_term + second_order_term
    control_variate.shape.assert_is_compatible_with([num_samples])

    return control_variate, second_order_gradients

  # Chain rule: we need to compute the gradients with respect to the mean,
  # and the gradients coming from samples.
  sample_gradients, _ = _cv(
      mean_dist, dist_samples, stop_grad_mean=True)
  param_gradients, second_order_gradients = _cv(
      mean_dist, tf.stop_gradient(dist_samples), stop_grad_mean=False)

  if grad_loss_fn:
    surrogate_cv, jacobians = grad_loss_fn(
        lambda x: _cv(mean_dist, x, stop_grad_mean=True)[0],
        dist_samples, dist)
    surrogate_cv += param_gradients

    # The second order expansion is done at the mean, so only the mean will
    # have jacobians here.
    # Note: we do not need to use parallel iterations here, since there is no
    # backpropagation through the samples.
    jacobians[dist.input_mean] += utils.jacobians(
        param_gradients, dist.input_mean)
  else:
    surrogate_cv = tf.zeros(())
    jacobians = {var: 0. for var in dist.dist_vars}

  expected_second_order_term = 1./2 * tf.reduce_sum(
      dist.variance() * tf.diag_part(second_order_gradients))

  posterior_mean_f = function(mean_dist)
  expected_control_variate = posterior_mean_f + expected_second_order_term

  return sample_gradients, expected_control_variate, surrogate_cv, jacobians


def moving_averages_baseline(
    dist, dist_samples, function, decay=0.9, grad_loss_fn=None):
  loss = tf.reduce_mean(function(dist_samples))
  moving_avg = tf.stop_gradient(snt.MovingAverage(decay=decay)(loss))
  control_variate = moving_avg
  expected_control_variate = moving_avg
  surrogate_cv, jacobians = grad_loss_fn(
      lambda x: moving_avg, dist_samples, dist)
  # Note: this has no effect on the gradient in the pathwise case.
  return control_variate, expected_control_variate, surrogate_cv, jacobians


def compute_control_variate_coeff(
    dist, dist_var, model_loss_fn, grad_loss_fn, control_variate_fn,
    num_samples, moving_averages=False, eps=1e-3):
  r"""Computes the control variate coefficients for the given variable.

  The coefficient is given by:
    \sum_k cov(df/d var_k, dcv/d var_k) / (\sum var(dcv/d var_k) + eps)

  Where var_k is the k'th element of the variable dist_var.
  The covariance and variance calculations are done from samples obtained
  from the distribution `dist`.

  Args:
    dist: a tfp.distributions.Distribution instance.
    dist_var: the variable for which we are interested in computing the
      coefficient.
      The distribution samples should depend on these variables.
    model_loss_fn: A function with signature: lambda samples: f(samples).
      The model loss function.
    grad_loss_fn: The gradient estimator function.
      Needs to return both a surrogate loss and a dictionary of jacobians.
    control_variate_fn: The surrogate control variate function. Its gradient
      will be used as a control variate.
    num_samples: Int. The number of samples to use for the cov/var calculation.
    moving_averages: Bool. Whether or not to use moving averages for the
      calculation.
    eps: Float. Used to stabilize division.

  Returns:
    a tf.Tensor of rank 0. The coefficient for the input variable.
  """
  # Resample to avoid biased gradients.

  cv_dist_samples = dist.sample(num_samples)
  cv_jacobians = control_variate_fn(
      dist, cv_dist_samples, model_loss_fn, grad_loss_fn=grad_loss_fn)[-1]
  loss_jacobians = grad_loss_fn(model_loss_fn, cv_dist_samples, dist)[-1]

  cv_jacobians = cv_jacobians[dist_var]
  loss_jacobians = loss_jacobians[dist_var]
  # Num samples x num_variables
  utils.assert_rank(loss_jacobians, 2)
  # Num samples x num_variables
  utils.assert_rank(cv_jacobians, 2)

  mean_f = tf.reduce_mean(loss_jacobians, axis=0)
  mean_cv, var_cv = tf.nn.moments(cv_jacobians, axes=[0])

  cov = tf.reduce_mean(
      (loss_jacobians - mean_f) * (cv_jacobians - mean_cv), axis=0)

  utils.assert_rank(var_cv, 1)
  utils.assert_rank(cov, 1)

  # Compute the coefficients which minimize variance.
  # Since we want to minimize the variances across parameter dimensions,
  # the optimal # coefficients are given by the sum of covariances per
  # dimensions over the sum of variances per dimension.
  cv_coeff = tf.reduce_sum(cov) / (tf.reduce_sum(var_cv) + eps)
  cv_coeff = tf.stop_gradient(cv_coeff)
  utils.assert_rank(cv_coeff, 0)
  if moving_averages:
    cv_coeff = tf.stop_gradient(snt.MovingAverage(decay=0.9)(cv_coeff))

  return cv_coeff


def control_variates_surrogate_loss(
    dist, dist_samples, dist_vars,
    model_loss_fn,
    grad_loss_fn,
    control_variate_fn,
    estimate_cv_coeff=True,
    num_posterior_samples_cv_coeff=20):
  r"""Computes a surrogate loss by computing the gradients manually.

  The loss function returned is:
     \sum_i stop_grad(grad_i) * var_i,
    where grad_i was computed from stochastic_loss and control variate.

  This function uses `compute_control_variate_coeff` to compute the control
  variate coefficients and should be used only in conjunction with control
  variates.

  Args:
    dist: a tfp.distributions.Distribution instance.
    dist_samples: samples from dist.
    dist_vars: the variables for which we are interested in computing gradients.
      The distribution samples should depend on these variables.
    model_loss_fn: A function with signature: lambda samples: f(samples).
      The model loss function.
    grad_loss_fn: The gradient estimator function.
      Needs to return both a surrogate loss and a dictionary of jacobians.
    control_variate_fn: The surrogate control variate function. Its gradient
      will be used as a control variate.
    estimate_cv_coeff: Boolean. Whether or not to use a coefficient
       for the control variate to minimize variance of the surrogate loss
       estimate. If False, the control variate coefficient is set to 1.
       If True, uses `compute_control_variate_coeff` to compute the coefficient.
    num_posterior_samples_cv_coeff: The number of posterior samples used
      to compute the cv coeff. Only used if `estimate_cv_coeff`
      is True.

  Returns:
    A tuple containing three elements:
      * the surrogate loss - a tf.Tensor [num_samples].
      * the jacobians wrt dist_vars.
      * a dict of debug information.
  """
  _, expected_control_variate, _, cv_jacobians = control_variate_fn(
      dist, dist_samples, model_loss_fn, grad_loss_fn=grad_loss_fn)

  _, loss_jacobians = grad_loss_fn(model_loss_fn, dist_samples, dist)
  jacobians = {}

  for dist_var in dist_vars:
    if estimate_cv_coeff:
      cv_coeff = compute_control_variate_coeff(
          dist, dist_var,
          model_loss_fn=model_loss_fn,
          grad_loss_fn=grad_loss_fn,
          control_variate_fn=control_variate_fn,
          num_samples=num_posterior_samples_cv_coeff)
    else:
      cv_coeff = 1.

    var_jacobians = loss_jacobians[dist_var] - cv_coeff * cv_jacobians[dist_var]
    # Num samples x num_variables
    utils.assert_rank(var_jacobians, 2)

    jacobians[dist_var] = var_jacobians

    utils.add_grads_to_jacobians(
        jacobians, expected_control_variate * cv_coeff, [dist_var])

  surrogate_loss = 0.0
  for dist_var in dist_vars:
    surrogate_loss += tf.stop_gradient(jacobians[dist_var]) * dist_var

  # Sum over variable dimensions.
  surrogate_loss = tf.reduce_sum(surrogate_loss, axis=1)

  return surrogate_loss, jacobians
