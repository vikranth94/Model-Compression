import tensorflow as tf
import time
import os

from model import Model
from configs import config

from data_util import data_loader, load_data

DATA_DIR = './cifar/'


def main():
    # load data
    ldDict ={}
    with open('labels.txt', 'r') as f:
        lines = f.readlines()
        for cid, line in enumerate(lines):
            labelname=line.split('\n')[0]
            ldDict[labelname] = cid
    labelNames = list(ldDict.keys())
    testSet = load_data(os.path.join(DATA_DIR, 'test/'), ldDict)
    print("Dataset loading completes.")

    # reset default graph
    tf.reset_default_graph()
    
    # create model
    model = Model(config, labelNames)
    # training setups
    saver = tf.train.Saver(max_to_keep=100)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        saver.restore(session, "50k1/model-50000")
        # test model
        test_accuracy = [0, 0]
        for step in range(10):
            test_idx = list(range(step*1000, step*1000+999))
            test_batch = {key: testSet[key][test_idx] for key in testSet}
            exp_results = run_single_step(session, model, test_batch, mode='test')
            test_accuracy[0] = test_accuracy[0] + exp_results['top_1_accuracy']
            test_accuracy[1] = test_accuracy[1] + exp_results['top_5_accuracy']
        print('top_1_accuracy_test = ', test_accuracy[0]/10, 'top_5_accuracy_test = ', test_accuracy[1]/10)


def run_single_step(
        session,
        model,
        batch,
        mode='test',
        log=True,
):
    # construct feed dict
    feed_dict = {
        model.images: batch['data'],
        # model.coarse_labels: batch['coarse_labels'],
        model.fine_labels: batch['labels'],
        # model.label_mapping: label_mapping,
        model.is_training: mode == 'train'
    }
    
    # select proper summary op
    if mode == 'train':
        summary_op = model.train_summary_op
    elif mode == 'val':
        summary_op = model.val_summary_op
    else:
        summary_op = model.test_summary_op
    
    # construct fetch list
    fetch_list = [model.global_step, summary_op, model.loss, model.accuracy, model.top_5_accuracy]

    # run single step
    _start_time = time.time()
    _step, _summary, _loss, _top_1, _top_5 = session.run(fetch_list, feed_dict=feed_dict)[:5]
    _end_time = time.time()
    
    # collect step statistics
    step_time = _end_time - _start_time
    batch_size = batch['data'].shape[0]
    
    # log in console
    if log:
        print(('[{:5s} step {:4d}] loss: {:.5f}; top_1_accuracy: {:.5f}; top_5_accuracy: {:5f} ' +
              '({:.3f} sec/batch; {:.3f} instances/sec)'
              ).format(mode, _step, _loss, _top_1, _top_5, 
                       step_time, batch_size / step_time))
    
    # log results to file and return statistics
    if mode == 'test':
        test_fetch_list = [model.per_class_accuracy,
                model.top_5_per_class_accuracy,
                model.confusion_matrix, 
                model.pred, model.probs]
        _top_1_c,  _top_5_c, _cm, _pred, _probs = \
                session.run(test_fetch_list, feed_dict=feed_dict)
        
        
        # Log detailed test results in pickle format
        stats = {
            "loss": _loss,
            "top_1_accuracy": _top_1,
            "top_5_accuracy": _top_5,
            "top_1_perclass_accuracy": _top_1_c,
            "top_5_perclass_accuracy": _top_5_c,
            "confusion_matrix": _cm,
            "pred": _pred,
            "probs": _probs
        }
    else:
        stats = {
            "step": _step,
            "loss": _loss,
            "top_1_accuracy": _top_1,
            "top_5_accuracy": _top_5
        }
        
    return stats


if __name__ == '__main__':
    main()
