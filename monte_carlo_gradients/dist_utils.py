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
"""Distribution utilities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp


tfd = tfp.distributions


def multi_normal(loc, log_scale):
  return MultiNormalDiagFromLogScale(loc=loc, log_scale=log_scale)


class MultiNormalDiagFromLogScale(tfd.MultivariateNormalDiag):
  """MultiNormalDiag which directly exposes its input parameters."""

  def __init__(self, loc, log_scale):
    scale = tf.exp(log_scale)
    self._log_scale = log_scale
    self._input_mean = loc
    super(MultiNormalDiagFromLogScale, self).__init__(
        loc, scale)

  @property
  def input_mean(self):
    return self._input_mean

  @property
  def log_scale(self):
    return self._log_scale

  @property
  def dist_vars(self):
    return [self.input_mean, self.log_scale]


def diagonal_gaussian_posterior(data_dims):
  mean = tf.Variable(
      tf.zeros(shape=(data_dims), dtype=tf.float32), name='mean')
  log_scale = tf.Variable(
      tf.zeros(shape=(data_dims), dtype=tf.float32), name='log_scale')
  return multi_normal(loc=mean, log_scale=log_scale)


def std_gaussian_from_std_dsmaxwell(std_dsmaxwell_samples):
  """Generate Gaussian variates from Maxwell variates.

  Useful for coupling samples from Gaussian and double_sided Maxwell dist.
  1. Generate ds-maxwell variates: dsM ~ dsMaxwell(0,1)
  2. Generate uniform variatres: u ~ Unif(0,1)
  3. multiply y = u * dsM
  The result is Gaussian distribution N(0,1) which can be loc-scale adjusted.

  Args:
    std_dsmaxwell_samples: Samples generated from a zero-mean, unit variance
    double-sided Maxwell distribution M(0,1).
  Returns:
    Tensor of Gaussian variates with shape maxwell_samples.
  """

  unif_rvs = tf.random.uniform(std_dsmaxwell_samples.shape)
  gaussian_rvs = unif_rvs * std_dsmaxwell_samples

  return gaussian_rvs


def sample_weibull(sh, scale, concentration):
  distrib = tfp.distributions.TransformedDistribution(
      distribution=tfp.distributions.Uniform(low=0., high=1. - 1e-6),
      bijector=tfp.bijectors.Invert(
          tfp.bijectors.Weibull(scale=scale, concentration=concentration)))
  return distrib.sample(sh)


def sample_ds_maxwell(sh, loc, scale):
  return tfd.DoublesidedMaxwell(loc=loc, scale=scale).sample(sh)
