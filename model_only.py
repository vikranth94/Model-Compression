import tensorflow as tf


# model starts here
def fcn(images, is_training=False, scope='fcn'):
    initializer = tf.contrib.layers.xavier_initializer()
    with tf.variable_scope(scope):
        _=images
        _ = tf.layers.conv2d(_, 96, (3, 3), 1, 'SAME',
                             activation=tf.nn.leaky_relu,
                             kernel_initializer=initializer,
                             name='conv1-1')
        _ = tf.layers.batch_normalization(_, training=is_training, name='norm1-1')
        _ = tf.layers.conv2d(_, 96, (3, 3), 1, 'SAME',
                             activation=tf.nn.leaky_relu,
                             kernel_initializer=initializer,
                             name='conv1-2')
        _ = tf.layers.batch_normalization(_, training=is_training, name='norm1-2')
        _ = tf.layers.max_pooling2d(_, (3, 3), 2, 'SAME',name='pool1')
        _ = tf.layers.conv2d(_, 192, (3, 3), 1, 'SAME',
                             activation=tf.nn.leaky_relu,
                             kernel_initializer=initializer,
                             name='conv2-1')
        _ = tf.layers.batch_normalization(_, training=is_training, name='norm2-1')
        _ = tf.layers.conv2d(_, 192, (3, 3), 1, 'SAME',
                             activation=tf.nn.leaky_relu,
                             kernel_initializer=initializer,
                             name='conv2-2')
        _ = tf.layers.batch_normalization(_, training=is_training, name='norm2-2')
        _ = tf.layers.max_pooling2d(_, (3, 3), 2, 'SAME', name='pool2')
        _ = tf.layers.conv2d(_, 192, (3, 3), 1, 'VALID',
                             activation=tf.nn.leaky_relu,
                             kernel_initializer=initializer,
                             name='conv3')
        _ = tf.layers.batch_normalization(_, training=is_training, name='norm3')
        _ = tf.layers.conv2d(_, 192, (1, 1), 1,
                             activation=tf.nn.leaky_relu,
                             kernel_initializer=initializer,
                             name='conv4')
        _ = tf.layers.batch_normalization(_, training=is_training, name='norm4')
        _ = tf.layers.conv2d(_, 10, (1, 1), 1,
                             activation=tf.nn.leaky_relu,
                             kernel_initializer=initializer,
                             name='conv5')
        _ = tf.layers.batch_normalization(_, training=is_training, name='norm5')
        _ = tf.layers.average_pooling2d(_, (6,6), 1, name='avg_pool')
        y = _
        logits = tf.reshape(y,[tf.shape(y)[0],10])
        return logits
