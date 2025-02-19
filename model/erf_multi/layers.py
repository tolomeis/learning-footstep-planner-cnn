import tensorflow as tf


class NonBottleneckResidualLayer(tf.keras.Model):

    def __init__(self, kernel_size, filters, strides, dilation=1, dropout=0.0, momentum=0.99, activation='elu'):
        super(NonBottleneckResidualLayer, self).__init__()

        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, (kernel_size[0], 1), (strides, 1), padding='same', kernel_initializer='he_normal'),
            tf.keras.layers.Activation(activation),
            tf.keras.layers.Conv2D(filters, (1, kernel_size[0]), (1, strides), padding='same', kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(momentum=momentum),
            tf.keras.layers.Activation(activation),
            tf.keras.layers.Conv2D(filters, (kernel_size[1], 1), dilation_rate=(dilation, 1), padding='same', kernel_initializer='he_normal'),
            tf.keras.layers.Activation(activation),
            tf.keras.layers.Conv2D(filters, (1, kernel_size[1]), dilation_rate=(1, dilation), padding='same', kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(momentum=momentum),
            tf.keras.layers.Dropout(dropout)
        ])
        self.activation = tf.keras.layers.Activation(activation)

    def call(self, inputs, training=None, mask=None):
        x = self.model(inputs, training=training)
        x = self.activation(x + inputs)
        return x


class DownSample(tf.keras.Model):

    def __init__(self, input_filters, output_filters, kernel_size, stride=2, momentum=0.99, activation='elu'):
        super(DownSample, self).__init__()
        self.conv = tf.keras.layers.Conv2D(output_filters - input_filters, kernel_size, stride, 'same', kernel_initializer='he_uniform')
        self.pool = tf.keras.layers.MaxPooling2D(stride, stride, 'same')
        self.activation = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(momentum=momentum),
            tf.keras.layers.Activation(activation)
        ])

    def call(self, inputs, training=None, mask=None):
        return self.activation(tf.concat([self.conv(inputs), self.pool(inputs)], -1), training=training)


class UpSample(tf.keras.Model):

    def __init__(self, kernel_size, filters, stride=2, momentum=0.99, activation='elu'):
        super(UpSample, self).__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(filters, kernel_size, stride, 'same', kernel_initializer='he_uniform'),
            tf.keras.layers.BatchNormalization(momentum=momentum),
            tf.keras.layers.Activation(activation)
        ])

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training=training)

class UpSampleScalar(tf.keras.Model):

    def __init__(self, kernel_size, filters, inpt_s, stride=2, momentum=0.99, activation='elu' ):
        super(UpSampleScalar, self).__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(filters, kernel_size, stride, 'same', kernel_initializer='he_uniform', input_shape = inpt_s),
            tf.keras.layers.BatchNormalization(momentum=momentum),
            tf.keras.layers.Activation(activation)
        ])

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training=training)