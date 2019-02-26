import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
print_tensors_in_checkpoint_file('../../tiandong/50k1/model-50000', all_tensors=False, all_tensor_names=False, tensor_name='')
