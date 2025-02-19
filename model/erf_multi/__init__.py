import tensorflow as tf

from .layers import DownSample, NonBottleneckResidualLayer, UpSample, UpSampleScalar


class ERF_M(tf.keras.Model):

    def __init__(self, output_channels, channels=3, momentum=0.9, dropout=0.3):
        super(ERF_M, self).__init__()
        kernel_size = [3, 3]
        self.num_classes = output_channels

        self.input_reshaper = tf.keras.layers.Reshape((40,40))
        

        self.encoder = tf.keras.Sequential([
            DownSample(channels, 16, 3, 2, momentum),
            DownSample(16, 64, 3, 2, momentum),
            NonBottleneckResidualLayer(kernel_size, 64, 1, 1, dropout, momentum),
            NonBottleneckResidualLayer(kernel_size, 64, 1, 1, dropout, momentum),
            NonBottleneckResidualLayer(kernel_size, 64, 1, 1, dropout, momentum),
            NonBottleneckResidualLayer(kernel_size, 64, 1, 1, dropout, momentum),
            NonBottleneckResidualLayer(kernel_size, 64, 1, 1, dropout, momentum),
            DownSample(64, 128, 3, 2, momentum),
            NonBottleneckResidualLayer(kernel_size, 128, 1, 2, dropout, momentum),
            NonBottleneckResidualLayer(kernel_size, 128, 1, 4, dropout, momentum),
            NonBottleneckResidualLayer(kernel_size, 128, 1, 8, dropout, momentum),
            NonBottleneckResidualLayer(kernel_size, 128, 1, 16, dropout, momentum),
            NonBottleneckResidualLayer(kernel_size, 128, 1, 2, dropout, momentum),
            NonBottleneckResidualLayer(kernel_size, 128, 1, 4, dropout, momentum),
            NonBottleneckResidualLayer(kernel_size, 128, 1, 8, dropout, momentum),
            NonBottleneckResidualLayer(kernel_size, 128, 1, 16, dropout, momentum)
        ])
        
        # Upscale the scalar input "foot shape" into a 5x5 
        self.shape_encoder = tf.keras.Sequential([
            tf.keras.layers.Reshape((1,1,1)),
            UpSampleScalar((5,5),1,(1,1,1), (5,5),momentum)
        ])

        self.decoder = tf.keras.Sequential([
            UpSample(3, 64, 2, momentum),
            NonBottleneckResidualLayer(kernel_size, 64, 1, 1, 0.0, momentum),
            NonBottleneckResidualLayer(kernel_size, 64, 1, 1, 0.0, momentum),
            UpSample(3, 16, 2, momentum),
            NonBottleneckResidualLayer(kernel_size, 16, 1, 1, 0.0, momentum),
            NonBottleneckResidualLayer(kernel_size, 16, 1, 1, 0.0, momentum)
        ])
        self.post_decoder = tf.keras.layers.Conv2DTranspose(output_channels, 2, 2, 'valid', kernel_initializer='he_uniform')

    def call(self, inputs, training=None): 
        # inputs starts as a vector, needs to be splitted and reshaped
        map = inputs[0]
        cp = inputs[1]
        cr = inputs[2]

        x = self.encoder(map, training=training)

        combined = tf.keras.layers.Concatenate()([x, \
            self.shape_encoder(cp, training = training), \
            self.shape_encoder(cr, training = training)])
        #combined = tf.concat([x, \
        #    self.shape_encoder(cp, training = training), \
        #    self.shape_encoder(cr, training = training)],-1)

        y = self.decoder(combined, training=training)
        return self.post_decoder(y)
        
        


    def get_model(self):
        #x = tf.keras.Input(shape=(40,40,1))
        #x = tf.keras.Input(shape=(40*40+1,1))

        #x = {"input_1":  tf.keras.Input(shape=(40,40,1)), "input_2":tf.keras.Input(shape=(2))}
        inp = [tf.keras.Input(shape=(40,40,1)), tf.keras.Input(shape=(1)), tf.keras.Input(shape=(1))]
        
        map = inp[0]
        cp = inp[1]
        cr = inp[2]

        x = self.encoder(map, training=False)
        m1 = tf.keras.Model(inputs = inp[0],outputs = x)

        combined = tf.keras.layers.concatenate()([x, \
            self.shape_encoder(cp, training = False), \
            self.shape_encoder(cr, training = False)])

        y = self.decoder(x, training=False)
        y = self.post_decoder(y)
        return tf.keras.Model(inputs=x,outputs=self.call(x))