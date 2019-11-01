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
"""Main experimental file."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import app
from absl import logging

import numpy as np
import tensorflow as tf

from monte_carlo_gradients import bayes_lr
from monte_carlo_gradients import blr_model_grad_utils
from monte_carlo_gradients import config
from monte_carlo_gradients import control_variates
from monte_carlo_gradients import data_utils
from monte_carlo_gradients import dist_utils
from monte_carlo_gradients import gradient_estimators


def _get_control_variate_fn():
  """Get control variate."""
  control_variate = config.gradient_config.control_variate
  if control_variate == 'delta':
    return control_variates.control_delta_method
  if control_variate == 'moving_avg':
    return control_variates.moving_averages_baseline

  if control_variate:
    raise ValueError('Unsupported control variate')

  return None


def _flag_checks():
  if (config.gradient_config.type != 'score_function' and
      config.control_variate == 'moving_avg'):
    raise ValueError(
        'Only the score function estimator supports a moving average baseline')


def _get_grad_loss_fn():
  """Get gradient loss function."""
  gradient_type_to_grad_loss_fn = {
      'pathwise': gradient_estimators.pathwise_loss,
      'score_function': gradient_estimators.score_function_loss,
      'measure_valued': gradient_estimators.measure_valued_loss,
  }

  return gradient_type_to_grad_loss_fn[config.gradient_config.type]


def _variance_reduction():
  return (config.gradient_config.control_variate or
          config.gradient_config.type == 'measure_valued')


def get_ckpt_dir(output_dir):
  ckpt_dir = os.path.join(output_dir, 'ckpt')
  if not tf.gfile.IsDirectory(ckpt_dir):
    tf.gfile.MakeDirs(ckpt_dir)
  return ckpt_dir


def _jacobian_parallel_iterations():
  # The pathwise estimator requires more memory since it uses the backward
  # pass through the model, so we use less parallel iterations to compute
  # jacobians.
  return 100 if config.gradient_config.type == 'pathwise' else None


def _configure_hooks(train_loss):
  checkpoint_saver_hook = tf.train.CheckpointSaverHook(
      checkpoint_dir=get_ckpt_dir(config.experiment_dir),
      save_steps=config.checkpoint_interval)

  nan_hook = tf.train.NanTensorHook(train_loss)
  # Step counter.
  step_conter_hook = tf.train.StepCounterHook()
  hooks = [checkpoint_saver_hook, nan_hook, step_conter_hook]
  return hooks


def _add_summaries(summary_writer, metrics):
  summary = tf.Summary()
  for name, value in metrics.items():
    summary.value.add(tag=name, simple_value=value)

  summary_writer.add_summary(summary, metrics['step'])
  summary_writer.flush()


def get_gradient_stats_for_logs(all_gradient_values):
  """Aggregate the stats for multiple mini batches of gradients."""
  # all_gradient_values is a list of dictionaries, containing the
  # jacobian values obtained at a particular iteration.
  # The keys of the dictionaries are the variables of the model.
  all_grads_list = {v: [] for v in all_gradient_values[0].keys()}
  for grad_values in all_gradient_values:
    for v, grads in grad_values.items():
      all_grads_list[v].append(grads)

  all_grads_np = {}
  for v, grads in all_grads_list.items():
    # Stack across batch dimension
    all_grads_np[v] = np.concatenate(all_grads_list[v], axis=0)

    # num_samples x num_params.
    assert len(all_grads_np[v].shape) == 2

  grad_stats = {}
  logged_keys = []
  for v, grads in all_grads_np.items():
    # Prefix the stats by the number of samples were used to compute them.
    num_eval_samples = grads.shape[0]
    key_prefix = str(num_eval_samples) + '_samples_'  + v
    # Get the mean and variance per dimension, reducing over number of samples.
    grad_mean = np.mean(grads, axis=0)
    grad_variance = np.var(grads, axis=0)

    # Get the mean and variance averaged over all dimensions.
    grad_stats[key_prefix + '_grad_global_mean'] = np.mean(grad_mean)

    # Max gradient variance across data dimensions.
    grad_stats[key_prefix + '_grad_across_dim_max_var'] = np.max(grad_variance)
    # Variance averaged across data dimensions.
    grad_stats[key_prefix + '_grad_var_mean'] = np.mean(grad_variance)

    # Estimate training variance by normalizing using the training number of
    # samples.
    num_training_samples = config.num_posterior_samples
    training_key_prefix = str(num_training_samples) + 'samples_'  + v

    # Var (1/n \sum_{i=0}^{n} X_i) = 1/n Var(X_i) since our estimates are iid.
    normalizing_factor = num_training_samples / num_eval_samples
    grad_stats[training_key_prefix + '_grad_across_dim_max_var'] = np.max(
        grad_variance) * normalizing_factor
    grad_stats[training_key_prefix + '_grad_var_mean'] = np.mean(
        grad_variance) * normalizing_factor

    logged_keys += [
        key_prefix + '_grad_across_dim_max_var', key_prefix + '_grad_var_mean']
  return grad_stats, logged_keys


def metrics_fetch_dict(model_output):
  return {
      'elbo': tf.reduce_mean(model_output.elbo),
      'kl': tf.reduce_mean(model_output.kl),
      'accuracy': model_output.accuracy}


def run_multi_batch_metrics(metrics, sess, num_eval_batches):
  stats = {k: 0 for k in metrics}
  for _ in range(num_eval_batches):
    metrics_values = sess.run(metrics)
    for k, v in metrics_values.items():
      stats[k] += v / num_eval_batches

  return stats


def run_gradient_stats(jacobians, sess, num_eval_batches):
  """Compute metrics on multiple batches."""
  all_jacobian_vals = []
  for _ in range(num_eval_batches):
    jacobian_vals = sess.run(jacobians)

    for v in jacobian_vals.values():
      # num_samples by num_params
      assert len(v.shape) == 2
    all_jacobian_vals += [jacobian_vals]

  gradient_stats, grad_log_keys = get_gradient_stats_for_logs(all_jacobian_vals)
  return gradient_stats, grad_log_keys


def _pretty_jacobians(jacobians):
  # Go from variable to jacobian to variable name to jacobian.
  return  {v.name: j for v, j in jacobians.items()}


def main(argv):
  del argv

  # Training data.
  features, targets = data_utils.get_sklearn_data_as_tensors(
      batch_size=config.batch_size,
      dataset_name=config.dataset_name)

  # Eval data.
  eval_features, eval_targets = data_utils.get_sklearn_data_as_tensors(
      batch_size=None,
      dataset_name=config.dataset_name)
  dataset_size = eval_features.get_shape()[0]

  data_dims = features.shape[1]

  prior = dist_utils.multi_normal(
      loc=tf.zeros(data_dims), log_scale=tf.zeros(data_dims))

  with tf.variable_scope('posterior'):
    posterior = dist_utils.diagonal_gaussian_posterior(data_dims)

  model = bayes_lr.BayesianLogisticRegression(
      prior=prior, posterior=posterior,
      dataset_size=dataset_size,
      use_analytical_kl=config.use_analytical_kl)

  grad_loss_fn = _get_grad_loss_fn()
  control_variate_fn = _get_control_variate_fn()
  jacobian_parallel_iterations = _jacobian_parallel_iterations()

  def model_loss(features, targets, posterior_samples):
    num_posterior_samples_cv_coeff = config.num_posterior_samples_cv_coeff
    return blr_model_grad_utils.model_surrogate_loss(
        model,
        features, targets, posterior_samples,
        grad_loss_fn=grad_loss_fn,
        control_variate_fn=control_variate_fn,
        estimate_cv_coeff=config.estimate_cv_coeff,
        num_posterior_samples_cv_coeff=num_posterior_samples_cv_coeff,
        jacobian_parallel_iterations=jacobian_parallel_iterations)

  posterior_samples = posterior.sample(config.num_posterior_samples)
  train_loss, _ = model_loss(features, targets, posterior_samples)
  train_loss = tf.reduce_mean(train_loss)

  num_eval_posterior_samples = config.num_eval_posterior_samples
  eval_posterior_samples = posterior.sample(num_eval_posterior_samples)
  eval_model_output = model.apply(
      eval_features, eval_targets, posterior_samples=eval_posterior_samples)

  _, jacobians = model_loss(
      eval_features, eval_targets, eval_posterior_samples)
  eval_model_metrics = metrics_fetch_dict(eval_model_output)
  jacobians = _pretty_jacobians(jacobians)

  # Compute the surrogate loss without any variance reduction.
  # Used as a sanity check and for debugging.
  if _variance_reduction():
    if control_variate_fn:
      no_var_reduction_grad_fn = grad_loss_fn
      no_var_reducion_prefix = 'no_control_variate'
    elif config.gradient_config.type == 'measure_valued':
      # Compute the loss and stats when not using coupling.
      def no_var_reduction_grad_fn(function, dist_samples, dist):
        return gradient_estimators.measure_valued_loss(
            function, dist_samples, dist, coupling=False)

    _, no_var_reduction_jacobians = blr_model_grad_utils.model_surrogate_loss(
        model, eval_features, eval_targets, eval_posterior_samples,
        grad_loss_fn=no_var_reduction_grad_fn,
        jacobian_parallel_iterations=jacobian_parallel_iterations)
    no_var_reduction_jacobians = _pretty_jacobians(no_var_reduction_jacobians)
    no_var_reducion_prefix = 'no_coupling'
  else:
    # No variance reduction used. No reason for additional logging.
    no_var_reduction_jacobians = {}

  for j in no_var_reduction_jacobians.values():
    assert j.get_shape().as_list()[0] == num_eval_posterior_samples

  start_learning_rate = config.start_learning_rate
  global_step = tf.train.get_or_create_global_step()

  if config.cosine_learning_rate_decay:
    training_steps = config.training_steps
    learning_rate_multiplier = tf.math.cos(
        np.pi / 2 * tf.cast(global_step, tf.float32)  / training_steps)
  else:
    learning_rate_multiplier = tf.constant(1.0)

  learning_rate = start_learning_rate * learning_rate_multiplier
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  train_op = optimizer.minimize(train_loss, global_step=global_step)

  hyper_dict = {
      'start_learning_rate': config.start_learning_rate,
      'num_posterior_samples': config.num_posterior_samples,
      'batch_size': config.batch_size}

  summary_writer = tf.summary.FileWriter(
      os.path.join(config.experiment_dir, 'logs'))

  # Checkpointing.
  hooks = _configure_hooks(train_loss)

  i = -1
  with tf.train.MonitoredSession(hooks=hooks) as sess:
    logging.info('starting training')
    for i in range(config.training_steps):
      sess.run(train_op)

      if (i + 1) % config.report_interval == 0:
        # Training loss and debug ops.
        logging.info('global_step %i', sess.run(global_step))
        logging.info('training loss at step %i: %f', i, sess.run(train_loss))

        # Compute multi batch eval metrics.
        multi_batch_metrics = run_multi_batch_metrics(
            eval_model_metrics, sess, config.num_eval_batches)

        for key, value in multi_batch_metrics.items():
          logging.info('%s at step %i: %f', key, i, value)

        posterior_vars_value = sess.run(
            {v.name: v for v in model.posterior.dist_vars})
        for k, value in posterior_vars_value.items():
          logging.info('%s avg at step %i: %f', k, i, np.mean(value))

        metrics = multi_batch_metrics
        metrics.update({'step': i})
        metrics.update({'learning_rate': sess.run(learning_rate)})
        metrics.update(hyper_dict)

        if (i + 1) % config.grad_report_interval == 0:
          gradient_stats, grad_log_keys = run_gradient_stats(
              jacobians, sess, config.num_eval_batches)

          for key in grad_log_keys:
            logging.info(
                '%s at step %i: %f', key, i, gradient_stats[key])
          metrics.update(gradient_stats)

          if no_var_reduction_jacobians:
            no_var_reduction_grad_stats, grad_log_keys = run_gradient_stats(
                no_var_reduction_jacobians, sess, config.num_eval_batches)

            no_var_reduction_grad_stats = {
                no_var_reducion_prefix + '_' + k: v
                for k, v in no_var_reduction_grad_stats.items()}

            metrics.update(no_var_reduction_grad_stats)

          _add_summaries(summary_writer, metrics)


if __name__ == '__main__':
  app.run(main)
