from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
assert tf.__version__.startswith('2')
import tensorflow.keras.backend as Keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Lambda

class ConditioningArgumentNetwork:
	def evaluate_conditional_variable(self, x):
		mean = x[:, :128]
		s = x[:, 128:]
		standard_deviation = tf.math.exp(s)
		exp = Keras.random_normal(shape=Keras.constant((mean.shape[1],), dtype='int32'))
		cond = mean + standard_deviation * exp
		return cond
	def build_ca_network(self):
		"""Structure of the conditioning augmentation network."""
		input1 = Input(shape=(1024,))
		cond_input = Dense(256)(input1)
		cond_input = LeakyReLU(alpha=0.2)(cond_input)
		condition_variable = Lambda(self.evaluate_conditional_variable)(cond_input)
		return Model(inputs=[input1], outputs=[condition_variable])
