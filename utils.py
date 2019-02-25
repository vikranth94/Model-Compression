import numpy as np
import os
from  natsort import natsorted
import imageio
import re
import time
import keras
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
import itertools

NAME = 'Cifar10_CNN'
data_dir = 'cifar'
model_dir = 'Models'
num_classes = 10

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship','truck']

class_dict = {
    'airplane': 0,
    'automobile':1,
    'bird':2,
    'cat':3,
    'deer':4,
    'dog':5,
    'frog':6,
    'horse':7,
    'ship':8,
    'truck':9
}

inv_class_dict = {v: k for k, v in class_dict.items()}


def prepare_dataset(data_dir, folder_name):
    try:
        print('Loading numpy')
        X = np.load('X_{}.npy'.format(folder_name))
        y = np.load('y_{}.npy'.format(folder_name))

    except:
        print('Loading images')
        image_list = []
        labels = []
        pictures_dir = os.path.join(data_dir, folder_name)
        names = [ d for d in os.listdir( pictures_dir ) if d.endswith( '.png') ]
        names = natsorted(names)
        for image in names:
            image_list.append(imageio.imread(os.path.join(pictures_dir, image)))
            label = re.split('[._]', image)
            labels.append(class_dict[label[1]])
            print(image)
        X = np.stack(image_list, axis=0)
        y = np.array(labels)
        np.save('X_{}'.format(folder_name),X)
        np.save('y_{}'.format(folder_name),y)
    return X,y

#z-score
def z_normalization(X, mean, std):
	X = (X-mean)/(std+1e-7)
	return X


def create_CNN_model(inp_shape, num_classes, p=0.2):
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),
	           activation='relu',
	           input_shape=inp_shape,
	           padding='same', name='Conv_1'))
	model.add(BatchNormalization(name='Bn_1')) 
	model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',padding='same',  name='Conv_2'))
	model.add(BatchNormalization(name='Bn_2')) 
	model.add(MaxPooling2D(pool_size=(2, 2), name='Max_pool_1'))  
	model.add(Dropout(p, name='Drop_1'))
	model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',padding='same',  name='Conv_3'))
	model.add(BatchNormalization(name='Bn_3')) 
	model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',padding='same',  name='Conv_4'))
	model.add(BatchNormalization(name='Bn_4')) 
	model.add(MaxPooling2D(pool_size=(2, 2), name='Max_pool_2'))  
	model.add(Dropout(p, name='Drop_2'))
	model.add(Conv2D(128, kernel_size=(3, 3), activation='relu',padding='same',  name='Conv_5'))
	model.add(BatchNormalization(name='Bn_5')) 
	model.add(Conv2D(128, kernel_size=(3, 3), activation='relu',padding='same',  name='Conv_6'))
	model.add(BatchNormalization(name='Bn_6')) 
	model.add(MaxPooling2D(pool_size=(2, 2), name='Max_pool_3'))  
	model.add(Dropout(p, name='Drop_3'))
	model.add(Flatten(name = 'Flatten_1'))
	model.add(Dense(32, activation='relu'))
	model.add(BatchNormalization(name='Bn_7')) 
	model.add(Dropout(p, name='Drop_4'))
	model.add(Dense(num_classes, activation='softmax', name='dense_out'))
	print(model.summary())
	return model


def train_CNN_model(model, X_train, y_train, X_val, y_val, model_dir, t, batch_size=256, epochs=50):
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    # checkpoint
    chk_path = os.path.join(model_dir, 'best_{}_{}'.format(NAME,t))
    checkpoint = ModelCheckpoint(chk_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    tensorboard = TensorBoard(log_dir="logs/{}_{}".format(NAME,t))
    callbacks_list = [checkpoint, tensorboard]

    history = model.fit(X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                shuffle=True,
                validation_data=(X_val, y_val),
                callbacks=callbacks_list)
    
    #Saving the model
    model.save(os.path.join(model_dir, 'final_{}_{}'.format(NAME,t)))
    return model, history


def calculate_metrics(model, X_test, y_test_binary):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test_binary, axis=1)
    mismatch = np.where(y_true != y_pred)
    cf_matrix = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    #micro_f1 = f1_score(y_true, y_pred, average='micro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    return cf_matrix, accuracy, macro_f1, mismatch, y_pred


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        print(cm)
    else:
        print('Confusion matrix, without normalization')
        print(cm)

    plt.figure(figsize = (10,7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize = 15)
    plt.yticks(tick_marks, classes, fontsize = 15)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 15,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    
    plt.ylabel('True label', fontsize = 12)
    plt.xlabel('Predicted label', fontsize = 12)