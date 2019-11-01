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
"""Tests for utility functions for distributions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import tensorflow_probability as tfp

from monte_carlo_gradients import dist_utils

tfd = tfp.distributions


class DistUtilsTest(tf.test.TestCase):

  def testGaussianfromMaxwellShape(self):
    sample_shape = (10, 5, 7)
    loc = tf.zeros(sample_shape)
    scale = tf.ones(sample_shape)
    n_samples = int(10e3)

    dsmaxwell = tfd.DoublesidedMaxwell(
        loc=loc, scale=scale)

    samples = dsmaxwell.sample(n_samples, seed=100)
    std_gaussian_rvs = dist_utils.std_gaussian_from_std_dsmaxwell(samples)

    self.assertEqual(std_gaussian_rvs.shape[1:], sample_shape)

  def testGaussianfromMaxwell(self):
    shape = (5, 10)
    mu = 3. * tf.ones(shape)
    sigma = 1.8 * tf.ones(shape)
    n_samples = int(10e3)

    dsmaxwell = tfd.DoublesidedMaxwell(
        loc=tf.zeros(shape), scale=tf.ones(shape))

    samples = dsmaxwell.sample(n_samples, seed=100)
    std_gaussian_rvs = (
        dist_utils.std_gaussian_from_std_dsmaxwell(samples))
    gaussian_rvs = std_gaussian_rvs*sigma + mu

    sample_mean = tf.reduce_mean(gaussian_rvs, 0)
    sample_variance = tf.sqrt(
        tf.reduce_mean(tf.square(gaussian_rvs - sample_mean), 0))

    self.assertAllClose(sample_mean, mu, rtol=0.05)
    self.assertAllClose(sample_variance, sigma, rtol=0.05)


if __name__ == '__main__':
  tf.test.main()
