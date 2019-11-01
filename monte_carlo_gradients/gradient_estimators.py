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
"""Implementation of gradient estimators.

The gradient estimators below have the same API:

````
 def grad_estimator(function, dist_samples, dist):
  return surrogate_loss, jacobian_dict

```

where `function` is a function that takes in as input a
`[num_samples, samples_dim]` tensor, and returns a tensor of size `num_samples`.
Thus, `function` has to be parallelizable across the first input dimension.
This can be achieved either via linear algebra operations, or by using
`snt.BatchApply`. `dist_samples` is a tensor of size
`[num_samples, samples_dim]` samples obtained from the `tfp.Distributions`
instance `dist`.

A gradient estimator needs to return a surrogate loss (a tensor of shape
`num_samples`), that can be used for optimization via automatic differentiation,
as well as a dictionary from a distribution variable of `dist` to the Jacobians
of the surrogate loss with respect to that variable. The Jacobians are used
to monitor variance and are not used for training.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import sonnet as snt
import tensorflow as tf

from monte_carlo_gradients import dist_utils
from monte_carlo_gradients import utils


def score_function_loss(function, dist_samples, dist):
  """Computes the score_function surrogate loss."""
  log_probs = dist.log_prob(tf.stop_gradient(dist_samples))

  # log_probs is of the size of the number of samples.
  utils.assert_rank(log_probs, 1)

  # Broadcast the log_probs to the loss.
  loss = tf.stop_gradient(function(dist_samples)) * log_probs
  return loss


def pathwise_loss(function, dist_samples, dist):
  """Computes the pathwise loss."""
  del dist
  return function(dist_samples)


def _apply_f(f, t):
  return snt.BatchApply(f, 2)(t)


def _measure_valued_normal_mean_grad(
    model_loss_fn, dist_samples, dist, coupling=True):
  """Computes the measure valued gradient wrt the mean of the Normal `dist`.

  For details, see Section 6 of
    "Monte Carlo Gradient Estimation in Machine learning".

  Args:
    model_loss_fn: A function for which to compute stochastic gradient for.
    dist_samples: a tf.Tensor of samples from `dist`.
    dist: A tfp.distributions.Distribution instance.
      The code here assumes this distribution is from the Normal family.
    coupling: A boolean. Whether or not to use coupling for the positive and
      negative samples. Recommended: True, as this  usually reduces variance.

  Returns:
    A tf.Tensor of size `num_samples`, where `num_samples` are the number
      of samples (the first dimension) of `dist_samples`. The gradient of
      model_loss_fn(dist_samples) wrt to the mean of `dist`.
  """
  mean = dist.loc
  # We will rely on backprop to compute the right gradient with respect
  # to the log scale.
  scale = dist.stddev()

  utils.assert_rank(mean, 1)
  utils.assert_rank(scale, 1)
  # Duplicate the D dimension - N x D x D.
  base_dist_samples = utils.tile_second_to_last_dim(dist_samples)

  shape = dist_samples.shape
  # N x D
  pos_sample = dist_utils.sample_weibull(
      shape, scale=tf.sqrt(2.), concentration=2.)
  pos_sample.shape.assert_is_compatible_with(shape)

  if coupling:
    neg_sample = pos_sample
  else:
    neg_sample = dist_utils.sample_weibull(
        shape, scale=tf.sqrt(2.), concentration=2.)

  neg_sample.shape.assert_is_compatible_with(shape)
  # N x D
  positive_diag = mean + scale * pos_sample
  positive_diag.shape.assert_is_compatible_with(shape)
  # N x D
  negative_diag = mean - scale * neg_sample
  negative_diag.shape.assert_is_compatible_with(shape)
  # Set the positive and negative - N x D x D
  positive = tf.linalg.set_diag(base_dist_samples, positive_diag)
  negative = tf.linalg.set_diag(base_dist_samples, negative_diag)

  c = np.sqrt(2 * np.pi) * scale  # D
  f = model_loss_fn
  # Broadcast the division.
  grads = (_apply_f(f, positive) - _apply_f(f, negative)) / c
  # grads - N x D
  grads.shape.assert_is_compatible_with(shape)

  return grads


def _measure_valued_normal_scale_grad(
    function, dist_samples, dist, coupling=True):
  """Computes the measure valued gradient wrt the `scale of the Normal `dist`.

  For details, see Section 6 of
    "Monte Carlo Gradient Estimation in Machine learning".

  Args:
    function: A function for which to compute stochastic gradient for.
    dist_samples: a tf.Tensor of samples from `dist`.
    dist: A tfp.distributions.Distribution instance.
      The code here assumes this distribution is from the Normal family.
    coupling: A boolean. Whether or not to use coupling for the positive and
      negative samples. Recommended: True, as this reduces variance.

  Returns:
    A tf.Tensor of size `num_samples`, where `num_samples` are the number
      of samples (the first dimension) of `dist_samples`. The gradient of
      function(dist_samples) wrt to the scale of `dist`.
  """
  mean = dist.loc
  # We will rely on backprop to compute the right gradient with respect
  # to the log scale.
  scale = dist.stddev()
  utils.assert_rank(mean, 1)
  utils.assert_rank(scale, 1)

  # Duplicate the D dimension - N x D x D
  base_dist_samples = utils.tile_second_to_last_dim(dist_samples)

  shape = dist_samples.shape
  # N x D
  pos_sample = dist_utils.sample_ds_maxwell(shape, loc=0., scale=1.0)
  if coupling:
    neg_sample = dist_utils.std_gaussian_from_std_dsmaxwell(pos_sample)
  else:
    neg_sample = tf.random.normal(shape)

  # N x D
  positive_diag = mean + scale * pos_sample
  positive_diag.shape.assert_is_compatible_with(shape)
  # N x D
  negative_diag = mean + scale * neg_sample
  negative_diag.shape.assert_is_compatible_with(shape)
  # Set the positive and negative values - N x D x D.
  positive = tf.linalg.set_diag(base_dist_samples, positive_diag)
  negative = tf.linalg.set_diag(base_dist_samples, negative_diag)

  c = scale  # D
  f = function
  # Broadcast the division.
  grads = (_apply_f(f, positive) - _apply_f(f, negative)) / c
  # grads - N x D
  grads.shape.assert_is_compatible_with(shape)

  return grads


def measure_valued_loss(
    function, dist_samples, dist, coupling=True):
  """Computes the surrogate loss via measure valued derivatives.

  Assumes `dist` is a Gaussian distribution.

  For details, see Section 6 of
    "Monte Carlo Gradient Estimation in Machine learning".

  Args:
    function: A function for which to compute stochastic gradient for.
    dist_samples: a tf.Tensor of samples from `dist`.
    dist: A tfp.distributions.Distribution instance.
      The code here assumes this distribution is from the Normal family.
    coupling: A boolean. Whether or not to use coupling for the positive and
      negative samples. Recommended: True, as this reduces variance.

  Returns:
    A tuple of two elements:
        1) tf.Tensor of size `num_samples`, where `num_samples` are the number
         of samples (the first dimension) of `dist_samples`. The surrogate loss
         that can be used as an input to an optimizer.
        2) A dictionary from distribution variable to Jacobians computed for
          that variable.

  """
  mean = dist.loc
  # We will rely on backprop to compute the right gradient with respect
  # to the log scale
  scale = dist.stddev()

  measure_valued_mean_grad = _measure_valued_normal_mean_grad(
      function, dist_samples, dist, coupling=coupling)

  measure_valued_scale_grad = _measure_valued_normal_scale_grad(
      function, dist_samples, dist, coupling=coupling)

  # Full surrogate loss which ensures the gradient wrt mean and scale are
  # the ones given by the measure valued derivatives.
  surrogate_loss = tf.stop_gradient(measure_valued_mean_grad) * mean
  surrogate_loss += tf.stop_gradient(measure_valued_scale_grad) * scale

  surrogate_loss.shape.assert_is_compatible_with(dist_samples.shape)
  # Reduce over the data dimensions.
  loss = tf.reduce_sum(surrogate_loss, axis=-1)

  # Create the gradients with respect to the log scale, since that is
  # what we plot in the other methods.
  log_scale_grads = measure_valued_scale_grad * scale

  jacobian_dict = {
      dist.input_mean: measure_valued_mean_grad,
      dist.log_scale: log_scale_grads}
  return loss, jacobian_dict
