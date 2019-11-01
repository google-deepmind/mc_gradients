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
"""Gradient utils."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from monte_carlo_gradients import control_variates
from monte_carlo_gradients import utils


def add_kl_to_loss(kl, loss, jacobian_dict):
  """Adds the KL to the optimized loss and updates jacobians accordingly."""
  if kl is None:
    return loss, jacobian_dict

  loss += kl
  jacobian_dict = utils.add_grads_to_jacobians(jacobian_dict, kl)

  return loss, jacobian_dict


def model_surrogate_loss(
    model, features, targets, posterior_samples, grad_loss_fn,
    control_variate_fn=None,
    estimate_cv_coeff=True,
    num_posterior_samples_cv_coeff=None,
    jacobian_parallel_iterations=None):
  r"""Computes the surrogate loss given the input model_outputs.

  It creates the appropriate surrogate function based on `grad_loss_fn`.

  The loss returned is of the form: \sum[stop_gradient(grad_var) * var].

  Args:
    model: An Instance of BayesianLogisticRegression.
    features: A tf.Tensor `[Batch size, data_dim]`.
    targets: A tf.Tensor `[Batch size]`. Values in `{0, 1}`.
    posterior_samples: A tf.Tensor `[Number of samples, data_dim]`.
    grad_loss_fn: The gradient estimator loss function.
    control_variate_fn: The control variate function.
    estimate_cv_coeff: Boolean. Whether or not to use a coefficient
       for the control variate to minimize variance of the surrogate loss
       estimate. If False, the control variate coefficient is set to 1.
    num_posterior_samples_cv_coeff: Integer. The number of samples used to
      compute the CV coeff.
    jacobian_parallel_iterations: None or Integer. The number of samples for
      which to compute the jacobian in parallel. Trades-off memory for speed.
  Returns:
    A tuple of size 2:
      * a tf.Tensor. The surrogate loss to optimize.
      * A dict from variable to the jacobians for the variable.
  """
  def model_loss_fn(x):
    return model.apply(features, targets, x).stochastic_batched_loss

  grad_loss_fn = utils.grad_loss_fn_with_jacobians(
      grad_loss_fn, jacobian_parallel_iterations=jacobian_parallel_iterations)

  if control_variate_fn:
    loss, jacobian_dict = control_variates.control_variates_surrogate_loss(
        dist=model.posterior,
        dist_samples=posterior_samples,
        dist_vars=model.posterior.dist_vars,
        model_loss_fn=model_loss_fn,
        grad_loss_fn=grad_loss_fn,
        control_variate_fn=control_variate_fn,
        estimate_cv_coeff=estimate_cv_coeff,
        num_posterior_samples_cv_coeff=num_posterior_samples_cv_coeff)
  else:
    loss, jacobian_dict = grad_loss_fn(
        function=model_loss_fn,
        dist=model.posterior,
        dist_samples=posterior_samples)

  loss, jacobian_dict = add_kl_to_loss(model.analytical_kl, loss, jacobian_dict)
  return loss, jacobian_dict
