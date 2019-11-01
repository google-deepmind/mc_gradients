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
"""Configuration parameters for main.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Config(dict):

  def __getattr__(self, key):
    return self[key]

  def __setattr__(self, key, value):
    self[key] = value


# Directory for the experiment logs, when running locally.
experiment_dir = '/tmp/tf_logs/'

# The dataset name.
dataset_name = 'breast_cancer'

# Training batch size.
batch_size = 32

# Use the entire dataset for evaluation.
eval_batch_size = None

# Number of batches to be used for evaluation.
num_eval_batches = 1

# The number of posterior samples used for evaluation.
num_eval_posterior_samples = 1000

# Number of posterior samples to be used for training.
num_posterior_samples = 50

# Initial learning rate.
start_learning_rate = 0.001

# Whether or not to use cosine learning rate decay.
cosine_learning_rate_decay = True

# Number of training steps.
training_steps = 5000

# Number of steps between loss logging.
report_interval = 10

# Number of steps between gradient logging.
grad_report_interval = 10

# Number of steps between checkpoints.
checkpoint_interval = 1000

gradient_config = Config()
# Options: 'pathwise', score_function', 'measure_valued'
gradient_config.type = 'pathwise'
gradient_config.control_variate = ''

# Whether or not use a control variate coefficient chosen
# to minimise the variance surrogate estimate.
# Set to False for experiments which control for the number of posterior
# samples used by the estimators.
estimate_cv_coeff = True
num_posterior_samples_cv_coeff = 25
# Whether or not to use the analytical KL computation.
use_analytical_kl = True

