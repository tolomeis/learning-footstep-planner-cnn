import numpy as np
import tensorflow as tf

from ..erf_multi import ERF_M

class MRF_generic(tf.keras.Model):
    def __init__(self, output_channels, channels, field_size, momentum, dropout):
        super(MRF_generic, self).__init__()

        self.erf = ERF_M(output_channels, channels, momentum, dropout)

        # k_internal - kernel for inside class probability distribution weighting
        # we check co-occurrence between each pair of labels globally
        # multiply by inverse eye because penalty for co-occurrence of the same classes is 0
        # shape: [classes, classes] -> [3, 3, classes, classes]
        self.k_internal = self.add_weight('k_internal', [field_size, field_size, output_channels, output_channels], tf.float32, initializer='zeros', trainable=True)
        self.k_internal_mask = tf.cast(~np.eye(output_channels, dtype=np.bool), tf.float32)[tf.newaxis, tf.newaxis]
        self.k_scale = tf.keras.layers.Conv2D(output_channels, 1, 1, 'same')

    def call(self, inputs, training=None, num_iters=1):
        x = self.erf(inputs, training=training)

        # calculate Q (softmax)
        q = tf.nn.softmax(x, -1)
        # get unary (high value for small probability because we calculate penalty)
        u = - tf.math.log(tf.clip_by_value(q, 1e-6, 1.0))

        logits = None
        for i in range(num_iters):
            internal_penalty = tf.nn.conv2d(q, self.k_internal * self.k_internal_mask, 1, 'SAME')
            penalty = internal_penalty  # + other penalties

            # get logits and scale for compatibility
            logits = - (u + penalty)
            logits = self.k_scale(logits)

            # calculate q, and go to next loop
            q = tf.nn.softmax(logits, -1)

        return logits

    def get_model(self, x):
        #x = tf.keras.Input(shape=(40,40,1))
        #x = tf.keras.Input(shape=(40*40+1,1))

        #x = {"input_1":  tf.keras.Input(shape=(40,40,1)), "input_2":tf.keras.Input(shape=(2))}
        
        return tf.keras.Model(inputs=x,outputs=self.call(x))

    def build_seq(self, map_shape, in_shape, latent_shape):
        self.erf.encoder.build(map_shape)
        self.erf.shape_encoder.build(in_shape)
        self.erf.decoder.build(latent_shape)

class MRF(tf.keras.Model):

    def __init__(self, output_channels, channels, field_size, momentum, dropout):
        super(MRF, self).__init__()

        self.erf = ERF(output_channels, channels, momentum, dropout)

        # k_internal - kernel for inside class probability distribution weighting
        # we check co-occurrence between each pair of labels globally
        # multiply by inverse eye because penalty for co-occurrence of the same classes is 0
        # shape: [classes, classes] -> [3, 3, classes, classes]
        self.k_internal = self.add_weight('k_internal', [field_size, field_size, output_channels, output_channels], tf.float32, initializer='zeros', trainable=True)
        self.k_internal_mask = tf.cast(~np.eye(output_channels, dtype=np.bool), tf.float32)[tf.newaxis, tf.newaxis]
        self.k_scale = tf.keras.layers.Conv2D(output_channels, 1, 1, 'same')

    def call(self, inputs, training=None, num_iters=1):
        x = self.erf(inputs, training=training)

        # calculate Q (softmax)
        q = tf.nn.softmax(x, -1)
        # get unary (high value for small probability because we calculate penalty)
        u = - tf.math.log(tf.clip_by_value(q, 1e-6, 1.0))

        logits = None
        for i in range(num_iters):
            internal_penalty = tf.nn.conv2d(q, self.k_internal * self.k_internal_mask, 1, 'SAME')
            penalty = internal_penalty  # + other penalties

            # get logits and scale for compatibility
            logits = - (u + penalty)
            logits = self.k_scale(logits)

            # calculate q, and go to next loop
            q = tf.nn.softmax(logits, -1)

        return logits
