import tensorflow as tf

flags = tf.app.flags

# experiment params
flags.DEFINE_string("prefix", "test_run", "Nickname for the experiment [default]")

# training params
flags.DEFINE_integer("max_steps", 50000, "Number of steps to train. [50000]")
flags.DEFINE_float("learning_rate", 1e-3, "Learning rate for the model. [1e-3]")
flags.DEFINE_integer("batch_size", 64, "Number of images in batch [64]")

# dataset params
flags.DEFINE_integer("num_classes", 10, "Number of classes in the dataset [100]")

# logging params
flags.DEFINE_integer("log_step", 500, "Interval for console logging [500]")
flags.DEFINE_integer("val_step", 500, "Interval for validation [5000]")
flags.DEFINE_integer("save_checkpoint_step", 5000, "Interval for checkpoint saving [5000]")


config = flags.FLAGS
