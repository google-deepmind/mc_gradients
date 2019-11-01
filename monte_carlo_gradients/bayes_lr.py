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
"""Bayesian logistic regression model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from absl import logging

import tensorflow as tf
import tensorflow_probability as tfp

from monte_carlo_gradients import utils


def _predictions(logits):
  return tf.to_float(logits >= 0)


def _accuracy(targets, predictions):
  # `predictions` have rank 2: batch_size, num_posterior samples.
  # We expand dims the targets to compute the accuracy per sample.
  targets = tf.expand_dims(targets, axis=1)
  return tf.reduce_mean(tf.to_float(tf.equal(targets, predictions)))


def linear_model(data, targets, posterior_samples):
  num_posterior_samples = tf.shape(posterior_samples)[0]
  logits = tf.matmul(data, posterior_samples, transpose_b=True)
  # Make targets [B, 1] to use broadcasting.
  targets = tf.expand_dims(targets, axis=1)
  targets = targets * tf.ones([1, num_posterior_samples])

  log_probs = - tf.nn.sigmoid_cross_entropy_with_logits(
      labels=targets, logits=logits)
  return log_probs, logits


ModelOutput = collections.namedtuple(
    'ModelOutput', ['elbo',
                    'predictions',
                    'accuracy',
                    'logits',
                    'log_probs',
                    'stochastic_batched_loss',
                    'kl',
                    'data_log_probs'])


class BayesianLogisticRegression(object):
  """Bayesian logistic regression model."""

  def __init__(
      self, prior, posterior,
      dataset_size,
      model_fn=linear_model,
      use_analytical_kl=True):
    self._prior = prior
    self._posterior = posterior
    self._dataset_size = tf.cast(dataset_size, tf.float32)
    self._use_analytical_kl = use_analytical_kl
    self._model_fn = model_fn

  @property
  def prior(self):
    return self._prior

  @property
  def posterior(self):
    return self._posterior

  def predict(self, data, targets, posterior_samples):
    _, logits = self.model_log_probs(
        data, targets, posterior_samples=posterior_samples)
    return _predictions(logits)

  @property
  def analytical_kl(self):
    """Computes the analytical KL."""
    if self._use_analytical_kl:
      try:
        kl = tfp.distributions.kl_divergence(self._posterior, self._prior)
      except NotImplementedError:
        logging.warn('Analytic KLD not available, using sampling KLD instead')
        self._use_analytical_kl = False

    if not self._use_analytical_kl:
      return None

    return kl

  def compute_kl(self, posterior_samples):
    """Computes the KL between the posterior to the prior."""
    if self._use_analytical_kl:
      return self.analytical_kl

    num_posterior_samples = posterior_samples.shape[0]
    # Compute the log probs of the posterior samples under the prior and
    # posterior. This is `num_posterior_samples`
    posterior_log_probs = self._posterior.log_prob(posterior_samples)
    posterior_log_probs.shape.assert_is_compatible_with(
        [num_posterior_samples])
    prior_log_probs = self._prior.log_prob(posterior_samples)
    prior_log_probs.shape.assert_is_compatible_with(
        [num_posterior_samples])

    kl = posterior_log_probs - prior_log_probs
    kl.shape.assert_is_compatible_with([num_posterior_samples])

    return kl

  def apply(self, features, targets, posterior_samples):
    """Applies the model and computes the elbo for the given inputs."""
    # Sample parameters from the posterior.
    batch_size = utils.get_shape_list(features)[0]
    num_posterior_samples = posterior_samples.shape[0]

    # Compute the log probs of the data under all the models we have sampled.
    log_probs, logits = self._model_fn(features, targets, posterior_samples)
    log_probs.shape.assert_is_compatible_with(
        [batch_size, num_posterior_samples])

    # The likelihood of the data is the average over parameters.
    data_log_probs = tf.reduce_mean(log_probs, axis=1)  # Reduce over samples.
    data_log_probs.shape.assert_is_compatible_with([batch_size])

    kl = self.compute_kl(posterior_samples)

    # Elbo computation - sum over data instances.
    param_log_probs = tf.reduce_mean(log_probs, axis=0) * self._dataset_size
    param_log_probs.shape.assert_is_compatible_with([num_posterior_samples])
    # Note: we rely on broadcasting to ensure the KL will be subtracted per
    # number of samples, in case the KL was computed analytically.
    elbo = param_log_probs - kl
    elbo.shape.assert_is_compatible_with([num_posterior_samples])

    predictions = _predictions(logits)
    accuracy = _accuracy(targets=targets, predictions=predictions)

    if self._use_analytical_kl:
      stochastic_batched_loss = - param_log_probs
    else:
      stochastic_batched_loss = - elbo

    stochastic_batched_loss.shape.assert_is_compatible_with(
        [num_posterior_samples])

    return ModelOutput(
        data_log_probs=data_log_probs,
        log_probs=log_probs,
        elbo=elbo,
        kl=kl,
        logits=logits,
        predictions=predictions,
        accuracy=accuracy,
        stochastic_batched_loss=stochastic_batched_loss)
