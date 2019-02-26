import tensorflow as tf


class Model(object):
    def __init__(self, config, label_names):
        self.config = config
        self.input_height = 32 
        self.input_width = 32 
        self.c_dim = 3 
        self.learning_rate = self.config.learning_rate
        self.num_classes = 10
        self.fine_label_names = label_names 
        self.images = tf.placeholder(name='images', dtype=tf.float32, shape=[None, self.input_height, self.input_width, self.c_dim])
        self.fine_labels = tf.placeholder(name='fine_labels', dtype=tf.int32, shape=[None])
        self.is_training = tf.placeholder(name='is_training', dtype=tf.bool, shape=[])
        self.build_model()
    
    def build_model(self):
        logits = self.fcn(self.images, is_training=self.is_training)
        # logits = self.resnet(self.images)
        self.probs = tf.nn.softmax(logits)
        self.pred = tf.cast(tf.argmax(logits, axis=1), tf.int32)
        
        # build loss
        self.loss = tf.losses.softmax_cross_entropy(tf.one_hot(self.fine_labels, self.num_classes), logits)
        
        # build optimizer ops
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.global_step = tf.train.get_or_create_global_step(graph=None)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optimizer_op = optimizer.minimize(self.loss, global_step=self.global_step)
        
        # metrics
        fine_labels_dn = tf.transpose(tf.one_hot(self.fine_labels, self.num_classes))
        top_1 = tf.cast(tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), self.fine_labels), tf.float32)
        self.accuracy = tf.reduce_mean(top_1)
        self.per_class_accuracy = tf.reduce_sum(fine_labels_dn * top_1, 1) / tf.reduce_sum(fine_labels_dn, 1)
        top_5 = tf.reduce_sum(tf.cast(tf.equal(tf.nn.top_k(logits, k=5, sorted=True)[1], 
                tf.expand_dims(self.fine_labels, 1)), tf.float32), axis=1)
        self.top_5_accuracy = tf.reduce_mean(top_5)
        self.top_5_per_class_accuracy = tf.reduce_sum(fine_labels_dn * top_5, 1) / tf.reduce_sum(fine_labels_dn, 1)
        self.confusion_matrix = tf.confusion_matrix(self.pred, self.fine_labels)
        
        # add summaries to Tensorboard
        self.add_summary("global_step", self.global_step)
        self.add_summary("loss", self.loss)
        self.add_summary("top_1/accuracy", self.accuracy)
        self.add_summary("top_5/accuracy", self.top_5_accuracy)
        for c in range(self.num_classes):
            self.add_summary("top_1/per_class_accuracy/{:03d}_{:s}" \
                             .format(c, self.fine_label_names[c]), self.per_class_accuracy[c])
            self.add_summary("top_5/per_class_accuracy/{:03d}_{:s}" \
                             .format(c, self.fine_label_names[c]), self.top_5_per_class_accuracy[c])

        # summaries
        self.train_summary_op = tf.summary.merge_all(key='train')
        self.val_summary_op = tf.summary.merge_all(key='val')
        self.test_summary_op = tf.summary.merge_all(key='test')

    # model starts here
    def fcn(self, images, is_training=False, scope='fcn'):
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

    def add_summary(self, name, value, summary_type='scalar'):
        if summary_type == 'scalar':
            tf.summary.scalar(name, value, collections=['train'])
            tf.summary.scalar('val_{}'.format(name), value, collections=['val'])
            tf.summary.scalar("test_{}".format(name), value, collections=['test'])
        elif summary_type == 'image':
            tf.summary.image(name, value, collections=['train'])
            tf.summary.image('val_{}'.format(name), value, collections=['val'])
            tf.summary.image("test_{}".format(name), value, collections=['test'])
