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
from scipy import special
import tensorflow as tf

from monte_carlo_gradients import bayes_lr
from monte_carlo_gradients import dist_utils


def _sigmoid(x):
  return special.expit(x)


class BayesianLogisticRegressionTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters([10, 12, 50])
  def testApplyZeroSamples(self, batch_size):
    data_dims = 10
    num_samples = 5
    dataset_size = 500

    mean = tf.Variable(
        tf.zeros(shape=(data_dims), dtype=tf.float32),
        name='mean')
    log_scale = tf.Variable(
        tf.zeros(shape=(data_dims), dtype=tf.float32),
        name='log_scale')

    # Prior = posterior.
    prior = dist_utils.multi_normal(loc=mean, log_scale=log_scale)
    posterior = dist_utils.multi_normal(loc=mean, log_scale=log_scale)
    model = bayes_lr.BayesianLogisticRegression(
        prior, posterior, dataset_size=dataset_size, use_analytical_kl=True)

    # Build the data
    features = tf.random.uniform((batch_size, data_dims))
    targets = tf.ones(batch_size)

    posterior_samples = tf.zeros((num_samples, data_dims))
    model_output = model.apply(
        features, targets, posterior_samples=posterior_samples)

    expected_predictions = np.ones((batch_size, num_samples))
    expected_accuracy = 1.

    expected_data_log_probs = np.log(0.5) * np.ones((batch_size))
    expected_elbo = np.log(0.5) * dataset_size * np.ones((num_samples))
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())

      self.assertEqual(sess.run(model.analytical_kl), 0)
      self.assertAllEqual(
          sess.run(model_output.predictions), expected_predictions)
      self.assertAllEqual(
          sess.run(model_output.accuracy), expected_accuracy)
      self.assertAllClose(
          sess.run(model_output.data_log_probs), expected_data_log_probs)
      self.assertAllClose(
          sess.run(model_output.elbo), expected_elbo)

  def testApply(self):
    data_dims = 10
    batch_size = 50
    num_samples = 6
    dataset_size = 500

    assert not batch_size % 2
    assert not num_samples % 2

    mean = tf.Variable(
        tf.zeros(shape=(data_dims), dtype=tf.float32),
        name='mean')
    log_scale = tf.Variable(
        tf.zeros(shape=(data_dims), dtype=tf.float32),
        name='log_scale')

    # Prior = posterior.
    prior = dist_utils.multi_normal(loc=mean, log_scale=log_scale)
    posterior = dist_utils.multi_normal(loc=mean, log_scale=log_scale)
    model = bayes_lr.BayesianLogisticRegression(
        prior, posterior, dataset_size=dataset_size, use_analytical_kl=True)

    # Build the data
    features = 3 * tf.ones((batch_size, data_dims), dtype=tf.float32)
    targets = tf.concat(
        [tf.zeros(int(batch_size/2), dtype=tf.float32),
         tf.ones(int(batch_size/2), dtype=tf.float32)],
        axis=0)

    posterior_samples = tf.concat(
        [tf.ones((int(num_samples/2), data_dims), dtype=tf.float32),
         -1 * tf.ones((int(num_samples/2), data_dims), dtype=tf.float32)],
        axis=0)

    model_output = model.apply(
        features, targets, posterior_samples=posterior_samples)

    expected_logits = 3 * data_dims * np.concatenate(
        [np.ones((batch_size, int(num_samples/2))),
         -1 * np.ones((batch_size, int(num_samples/2)))],
        axis=1)

    quarter_ones = np.ones((int(batch_size/2), int(num_samples/2)))
    # Compute log probs for the entire batch, for the first half of samples.
    first_half_data_expected_log_probs = np.concatenate(
        [np.log(1 - _sigmoid(3 * data_dims)) * quarter_ones,
         np.log(_sigmoid(3 * data_dims)) * quarter_ones], axis=0)

    # Compute log probs for the entire batch, for the second half of samples.
    second_half_data_expected_log_probs = np.concatenate(
        [np.log(1 - _sigmoid(- 3 * data_dims)) * quarter_ones,
         np.log(_sigmoid(- 3 * data_dims)) * quarter_ones], axis=0)

    expected_log_probs = np.concatenate(
        [first_half_data_expected_log_probs,
         second_half_data_expected_log_probs], axis=1)

    first_half_expected_elbo = np.log(1 - _sigmoid(3 * data_dims))
    first_half_expected_elbo += np.log(_sigmoid(3 * data_dims))

    second_half_expected_elbo = np.log(_sigmoid(-3 * data_dims))
    second_half_expected_elbo += np.log(1 - _sigmoid(-3 * data_dims))

    expected_elbo = dataset_size/2. * np.concatenate([
        first_half_expected_elbo* np.ones((int(num_samples/2))),
        second_half_expected_elbo * np.ones((int(num_samples/2)))])

    expected_predictions = np.concatenate(
        [np.ones((batch_size, int(num_samples/2))),
         np.zeros((batch_size, int(num_samples/2)))],
        axis=1)

    expected_accuracy = 0.5

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())

      self.assertEqual(sess.run(model_output.kl), 0)
      self.assertAllEqual(
          sess.run(model_output.logits), expected_logits)

      self.assertAllEqual(
          sess.run(model_output.predictions), expected_predictions)
      self.assertAllEqual(
          sess.run(model_output.accuracy), expected_accuracy)

      self.assertAllClose(
          sess.run(model_output.log_probs),
          expected_log_probs, rtol=1e-1, atol=5e-3)

      self.assertAllClose(
          sess.run(model_output.elbo), expected_elbo, rtol=1e-1, atol=5e-3)


if __name__ == '__main__':
  tf.test.main()
