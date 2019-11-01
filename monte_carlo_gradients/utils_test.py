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

# Dependency imports
import numpy as np
import tensorflow as tf

from monte_carlo_gradients import utils


class UtilsTest(tf.test.TestCase):

  def testRepeatLastDim(self):
    a = tf.constant([[0., 1.,], [2., 3.]])
    b = utils.tile_second_to_last_dim(a)

    with tf.Session() as sess:
      b_val = sess.run(b)

      self.assertAllEqual(b_val[0], np.array([[0., 1.], [0., 1.]]))
      self.assertAllEqual(b_val[1], np.array([[2., 3.], [2., 3.]]))


if __name__ == '__main__':
  tf.test.main()
