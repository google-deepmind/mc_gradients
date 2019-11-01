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
"""Data utilies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
from sklearn import datasets
import tensorflow as tf


def get_sklearn_data(dataset_name='breast_cancer', normalize=True):
  """Read sklearn datasets as numpy array."""
  if dataset_name == 'breast_cancer':
    data = datasets.load_breast_cancer()
  else:
    raise ValueError('Unsupported dataset')

  features = np.array(data.data, dtype=np.float32)
  targets = np.array(data.target, dtype=np.float32)

  if normalize:
    # Note: the data dimensions have very different scales.
    # We normalize by the mean and scale of the entire dataset.
    features = features - features.mean(axis=0)
    features = features / features.std(axis=0)

  assert targets.min() == 0
  assert targets.max() == 1

  # Add a dimension of 1 (bias).
  features = np.concatenate((features, np.ones((features.shape[0], 1))), axis=1)
  features = np.array(features, dtype=np.float32)

  return features, targets


def get_sklearn_data_as_tensors(batch_size=None, dataset_name='breast_cancer'):
  """Read sklearn datasets as tf.Tensors.

  Args:
    batch_size: Integer or None. If None, the entire dataset is used.
    dataset_name: A string, the name of the dataset.

  Returns:
    A tuple of size two containing two tensors of rank 2 `[B, F]`, the data
    features and targets.
  """
  features, targets = get_sklearn_data(dataset_name=dataset_name)
  dataset = tf.data.Dataset.from_tensor_slices((features, targets))
  if batch_size:
    # Shuffle, repeat, and batch the examples.
    batched_dataset = dataset.shuffle(1000).repeat().batch(batch_size)
  else:
    batch_size = features.shape[0]
    batched_dataset = dataset.repeat().batch(batch_size)

  iterator = batched_dataset.make_one_shot_iterator()
  batch_features, batch_targets = iterator.get_next()

  data_dim = features.shape[1]
  batch_features.set_shape([batch_size, data_dim])
  batch_targets.set_shape([batch_size])

  return batch_features, batch_targets

