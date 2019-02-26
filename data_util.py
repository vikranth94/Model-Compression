import os
import numpy as np
import cv2

"""

Loads and preprocesses cifar 10 data from file.

Args:
    dir: directory that includes the cifar 10 data.
Returns:
    3 dictionaries (train_data, val_data, test_data),
    each with two keys "data", and "label".
"""


def data_loader(dir):
    ldDict ={}
    with open(dir+'labels.txt', 'r') as f:
        lines = f.readlines()
        for cid, line in enumerate(lines):
            labelname=line.split('\n')[0]
            ldDict[labelname] = cid
    trainSet= load_data(os.path.join(dir, 'train/'), ldDict)
    valSet, trainSet_s = split_val_from_train(trainSet)
    testSet = load_data(os.path.join(dir,'test/'), ldDict)
    return trainSet,valSet,testSet, list(ldDict.keys())


def load_data(dirname, ldDict):
    data=[]
    labels=[]
    files = os.listdir(dirname)
    for filename in files:
        img = cv2.imread(dirname+filename)
        labelname = filename.split('.')[0].split('_')[1]
        cid = ldDict[labelname]
        data.append(img)
        labels.append(cid)
    dataSet = (np.asarray(data)/255.0).astype(np.float32)
    labels = np.asarray(labels)
    trainSet = {'data':dataSet,'labels':labels}
    return trainSet


def split_val_from_train(trainSet):
    N = trainSet['data'].shape[0]
    perm = np.random.permutation(N)
    val_idx = perm[int(N*0.8):]
    valSet = {key: trainSet[key][val_idx] for key in trainSet.keys()}
    train_idx = perm[:int(N*0.8)]
    trainSet_s = {key: trainSet[key][train_idx] for key in trainSet.keys()}
    return valSet, trainSet_s