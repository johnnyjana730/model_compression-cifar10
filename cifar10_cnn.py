from __future__ import print_function
import keras
from keras.datasets import cifar10
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import backend as K
from mobilenet import get_mobilenet
import h5py
import numpy as np
import os
from matplotlib import pyplot as plt
SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))

def show(img):
    io.imshow(img)
    io.show()
def softmax_c(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] 
    return e_x / div

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 32, 32

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# if K.image_data_format() == 'channels_first':
#     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(64, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 3)))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(5, 5),
                 activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(96, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.summary()
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(x_test, y_test))
# score = model.evaluate(x_test, y_test, verbose=0)

# model.save_weights(SCRIPT_PATH+'/model_1/cifar10_model_weights.h5')
model.load_weights(SCRIPT_PATH+'/model_1/cifar10_model_weights.h5')

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

convparams = 4864 + 256 + 102464 + 256
trainable_params = model.count_params()
footprint = trainable_params * 4
print ("Memory footprint per Image Feed Forward ~= " , footprint / 1024.0 /1024.0 ,"Mb")


from keras import backend as K
def extract_feature():
    def prepare_softtargets(model,X):
        inp = model.input                                     
        outputs = []
        for layer in model.layers[:]:
            if layer.name == 'flatten_1':
                outputs.append(layer.output)
            if layer.name == 'dense_1':
                outputs.append(layer.output)
            if layer.name == 'dense_2':
                outputs.append(layer.output)
            if layer.name == 'dense_3':
                outputs.append(layer.output)
        functor = K.function([inp]+ [K.learning_phase()], outputs ) # evaluation function
        layer_outs = functor([X, 1.])
        return np.array(layer_outs[0]) , np.array(layer_outs[1]), np.array(layer_outs[2]), np.array(layer_outs[3])

    lastconv_out = []
    logit_out = []
    logit_out2 = []
    logit_out3 = []
    for i in range(0,50):
        print ("Batch # : ",i)
        f1,d1,d2, d3 =  (prepare_softtargets(model,x_train[i*1000:(i+1)*1000]))
        print(f1.shape)
        lastconv_out.append(f1)
        logit_out.append(d1)
        logit_out2.append(d2)
        logit_out3.append(d3)
        #lastconv_out , logit_out = prepare_softtargets(model,x_train[i*1000:(i+1)*1000])

    # lastconv_out.shape , logit_out.shape
    lastconv_out = np.array(lastconv_out)
    logit_out = np.array(logit_out)
    logit_out2 = np.array(logit_out2)
    logit_out3 = np.array(logit_out3)
    print(lastconv_out.shape)
    lastconv_out = lastconv_out.reshape((50000 , 1600))
    logit_out = logit_out.reshape((50000 , 256))
    logit_out2 = logit_out2.reshape((50000 , 96))
    logit_out3 = logit_out3.reshape((50000 , 10))
    print("lastconv_out = ",lastconv_out.shape,"logit_out = ",logit_out.shape,"logit_out2 = ",logit_out2.shape,"logit_out3 = ",logit_out3.shape)

    print("clean up ")
    x_train = 0

    h5f = h5py.File(SCRIPT_PATH+'/model_1/lastconv_out.h5', 'w')
    h5f.create_dataset('dataset_1', data=lastconv_out)
    h5f.close()

    h5f2 = h5py.File(SCRIPT_PATH+'/model_1/logit_out.h5', 'w')
    h5f2.create_dataset('dataset_1', data=logit_out)
    h5f2.close()

    h5f3 = h5py.File(SCRIPT_PATH+'/model_1/logit_out2.h5', 'w')
    h5f3.create_dataset('dataset_1', data=logit_out2)
    h5f3.close()

    h5f4 = h5py.File(SCRIPT_PATH+'/model_1/logit_out3.h5', 'w')
    h5f4.create_dataset('dataset_1', data=logit_out3)
    h5f4.close()

    # free up memory
    lastconv_out = 0
    logit_out = 0 
    logit_out2 = 0
    logit_out3 = 0

    # test data set
    lastconv_out = []
    logit_out = []
    logit_out2 = []
    logit_out3 = []

    for i in range(0,10):
        print ("Batch # : ",i)
        f1,d1,d2, d3 =  ( prepare_softtargets(model,x_test[i*1000:(i+1)*1000]))
        lastconv_out.append(f1)
        logit_out.append(d1)
        logit_out2.append(d2)
        logit_out3.append(d3)
        #lastconv_out , logit_out = prepare_softtargets(model,x_train[i*1000:(i+1)*1000])

    # lastconv_out.shape , logit_out.shape
    lastconv_out = np.array(lastconv_out)
    logit_out = np.array(logit_out)
    logit_out2 = np.array(logit_out2)
    logit_out3 = np.array(logit_out3)
    lastconv_out = lastconv_out.reshape((10000 , 1600))
    logit_out = logit_out.reshape((10000 , 256))
    logit_out2 = logit_out2.reshape((10000 , 96))
    logit_out3 = logit_out3.reshape((10000 , 10))


    h5f = h5py.File(SCRIPT_PATH+'/model_1/test_lastconv_out.h5', 'w')
    h5f.create_dataset('dataset_1', data=lastconv_out)
    h5f.close()

    h5f2 = h5py.File(SCRIPT_PATH+'/model_1/test_logit_out.h5', 'w')
    h5f2.create_dataset('dataset_1', data=logit_out)
    h5f2.close()

    h5f3 = h5py.File(SCRIPT_PATH+'/model_1/test_logit_out2.h5', 'w')
    h5f3.create_dataset('dataset_1', data=logit_out2)
    h5f3.close()

    h5f4 = h5py.File(SCRIPT_PATH+'/model_1/test_logit_out3.h5', 'w')
    h5f4.create_dataset('dataset_1', data=logit_out3)
    h5f4.close()

def plot_findings(results,name,save=False):
    fig = plt.figure()
    fig.suptitle('parameter Size and Accuracy', fontsize=20)
    plt.plot([r['nparams_student'] for r in results] , [r['accuracy_student'] for r in results])
    plt.xlabel('Parameter Size', fontsize=18)
    plt.ylabel('accuracy', fontsize=16)
    if save:  fig.savefig(SCRIPT_PATH+'/model_1/plots/' + name + '_parameterSize_Accuracy.png')
    # plt.show()
    fig = plt.figure()
    fig.suptitle('Compression Rate and Accuracy', fontsize=20)
    plt.plot([r['compressionRate'] for r in results] , [r['accuracy_student'] for r in results])
    plt.xlabel('compression Rate', fontsize=18)
    plt.ylabel('accuracy', fontsize=18)
    if save:  fig.savefig(SCRIPT_PATH+'/model_1/plots/' + name +'_CompressionRate_Accuracy.png')
    # plt.show()

def finetunedense1_dense2():
    # finetunedense1_dense2
    print('finetunedense1_dense2')
    results = []
    for HiddenNeuron in range(2,20,3):

        h5f = h5py.File(SCRIPT_PATH+'/model_1/lastconv_out.h5', 'r')
        lastconv_out = h5f['dataset_1'][:]
        h5f.close()

        h5f4 = h5py.File(SCRIPT_PATH+'/model_1/logit_out3.h5', 'r')
        logit_out3 = h5f4['dataset_1'][:]
        h5f4.close()

        student_model = Sequential()
        student_model.add(Dense(HiddenNeuron*2,input_dim=1600,activation='relu'))
        student_model.add(Dense(HiddenNeuron, activation='relu'))
        student_model.add(Dense(num_classes, activation='softmax'))

        student_model.compile(loss='mse',
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])

        student_model.fit(lastconv_out, logit_out3,nb_epoch=40,verbose=0)
    #     student_model.save_weights("student_weights_"+str(HiddenNeuron)+"hidden_0.5_dropout.h5")
        
        # Compression Rate from Number of Parameters Reduced
        dense2_dense1_para = 409856 + 24672

        print("dense1 and dense2 fine tune")
        print ("HiddenNeurons : " , HiddenNeuron)
        print ("Initial Parameters : " , model.count_params())
        print ("Compressed parameters: ", model.count_params() - dense2_dense1_para + student_model.count_params())
        compressionRate = model.count_params() / np.float(model.count_params() - dense2_dense1_para + student_model.count_params())
        print ("Compression Rate : " , compressionRate)
        
        lastconv_out = 0
        logit_out2 = 0                

        h5f = h5py.File(SCRIPT_PATH+'/model_1/test_lastconv_out.h5', 'r')
        test_lastconv_out = h5f['dataset_1'][:]
        h5f.close()

        h5f4 = h5py.File(SCRIPT_PATH+'/model_1/test_logit_out3.h5', 'r')
        test_logit_out = h5f4['dataset_1'][:]
        h5f4.close()
        
        pred = student_model.predict(test_lastconv_out)
        probs = softmax_c(pred)
        pred_classes = np.argmax(probs,axis=1)

        accuracy_student = metrics.accuracy_score(y_pred=pred_classes,y_true=np.argmax(y_test,axis=1))
        print ("Accuracy compare with test set: " , accuracy_student)
                                
        out = {
            "HiddenNeuron" :    HiddenNeuron,
            "compressionRate" : compressionRate,
            "nparams_student" : student_model.count_params()  + convparams,
            "accuracy_student": accuracy_student
        }
        
        student_model = 0 
        lastconv_out = 0
        logit_out = 0 
        test_lastconv_out = 0
        test_logit_out = 0 
        results.append(out)

    print('finetunedense1_dense2 = ',results)

    plot_findings(results,'dense1_dense2',save=True)

def remove_d1_d2():
    # remove_d1_d2
    print('remove_d1_d2')
    results = []
    for HiddenNeuron in range(2,15,3):

        h5f = h5py.File(SCRIPT_PATH+'/model_1/lastconv_out.h5', 'r')
        lastconv_out = h5f['dataset_1'][:]
        h5f.close()

        h5f4 = h5py.File(SCRIPT_PATH+'/model_1/logit_out3.h5', 'r')
        logit_out3 = h5f4['dataset_1'][:]
        h5f4.close()

        student_model = Sequential()
        student_model.add(Dense(HiddenNeuron,input_dim=1600,activation='relu'))
        student_model.add(Dense(num_classes, activation='softmax'))

        student_model.compile(loss='mse',
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])

        student_model.fit(lastconv_out, logit_out3,nb_epoch=40,verbose=0)
    #     student_model.save_weights("student_weights_"+str(HiddenNeuron)+"hidden_0.5_dropout.h5")
        
        # Compression Rate from Number of Parameters Reduced
        dense2_dense1_para = 409856 + 24672

        print("dense1 and dense2 fine tune")
        print ("HiddenNeurons : " , HiddenNeuron)
        print ("Initial Parameters : " , model.count_params())
        print ("Compressed parameters: ", model.count_params() - dense2_dense1_para + student_model.count_params())
        compressionRate = model.count_params() / np.float(model.count_params() - dense2_dense1_para + student_model.count_params())
        print ("Compression Rate : " , compressionRate)
        
        lastconv_out = 0
        logit_out2 = 0                

        h5f = h5py.File(SCRIPT_PATH+'/model_1/test_lastconv_out.h5', 'r')
        test_lastconv_out = h5f['dataset_1'][:]
        h5f.close()

        h5f4 = h5py.File(SCRIPT_PATH+'/model_1/test_logit_out3.h5', 'r')
        test_logit_out = h5f4['dataset_1'][:]
        h5f4.close()
        
        pred = student_model.predict(test_lastconv_out)
        probs = softmax_c(pred)
        pred_classes = np.argmax(probs,axis=1)

        accuracy_student = metrics.accuracy_score(y_pred=pred_classes,y_true=np.argmax(y_test,axis=1))
        print ("Accuracy compare with test set: " , accuracy_student)
                                
        out = {
            "HiddenNeuron" :    HiddenNeuron,
            "compressionRate" : compressionRate,
            "nparams_student" : student_model.count_params()  + convparams,
            "accuracy_student": accuracy_student
        }
        
        student_model = 0 
        lastconv_out = 0
        logit_out = 0 
        test_lastconv_out = 0
        test_logit_out = 0 
        
        results.append(out)

    print('remove_d1_d2 = ', results)

    plot_findings(results,'remove_d1_d2',save=True)

def plot_findings2(results,name,save=False):
    fig = plt.figure()
    fig.suptitle('parameter Size and dense2 Accuracy', fontsize=20)
    plt.plot([r['nparams_student'] for r in results] , [r['accuracy_student_premodel_prediction'] for r in results])
    plt.xlabel('Parameter Size', fontsize=18)
    plt.ylabel('accuracy', fontsize=16)
    if save:  fig.savefig(SCRIPT_PATH+'/model_1/plots/' + name + '_parameterSize_Accuracy.png')
    # plt.show()
    fig = plt.figure()
    fig.suptitle('Compression Rate and dense2 Accuracy', fontsize=20)
    plt.plot([r['compressionRate'] for r in results] , [r['accuracy_student_premodel_prediction'] for r in results])
    plt.xlabel('compression Rate', fontsize=18)
    plt.ylabel('accuracy', fontsize=18)
    if save:  fig.savefig(SCRIPT_PATH+'/model_1/plots/' + name +'_CompressionRate_Accuracy.png')
    # plt.show()

def finetunedense2():
    # finetunedense2
    print('finetunedense2')
    results = []
    for HiddenNeuron in range(96,256,30):

        h5f = h5py.File(SCRIPT_PATH+'/model_1/lastconv_out.h5', 'r')
        lastconv_out = h5f['dataset_1'][:]
        h5f.close()

        h5f3 = h5py.File(SCRIPT_PATH+'/model_1/logit_out2.h5', 'r')
        logit_out2 = h5f3['dataset_1'][:]
        h5f3.close()

        student_model = Sequential()
        student_model.add(Dense(HiddenNeuron,input_dim=1600,activation='relu'))
        student_model.add(Dense(96, activation='relu'))

        student_model.compile(loss='mse',
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])

        student_model.fit(lastconv_out, logit_out2,nb_epoch=40,verbose=0)
    #     student_model.save_weights("student_weights_"+str(HiddenNeuron)+"hidden_0.5_dropout.h5")
        
        # Compression Rate from Number of Parameters Reduced
        dense2_para = 409856

        print("dense2 fine tune")
        print ("HiddenNeurons : " , HiddenNeuron)
        print ("Initial  Parameters : " , model.count_params())
        print ("Compressed dense2 parameters: ", model.count_params() - dense2_para + student_model.count_params())
        compressionRate = model.count_params() / np.float(model.count_params() - dense2_para + student_model.count_params())
        print ("Compression Rate : " , compressionRate)
        
        lastconv_out = 0
        logit_out2 = 0                

        h5f = h5py.File(SCRIPT_PATH+'/model_1/test_lastconv_out.h5', 'r')
        test_lastconv_out = h5f['dataset_1'][:]
        h5f.close()

        h5f3 = h5py.File(SCRIPT_PATH+'/model_1/test_logit_out2.h5', 'r')
        test_logit_out = h5f3['dataset_1'][:]
        h5f3.close()
        
        accuracy_student = student_model.evaluate(test_lastconv_out, test_logit_out)
        # accuracy_student = rfr.score(y_pred=np.array(pred),y_true=np.array(test_logit_out))
        print ("Accuracy compare with previous model prediction: " , accuracy_student[1])
                                
        out = {
            "HiddenNeuron" :    HiddenNeuron,
            "compressionRate" : compressionRate,
            "nparams_student" : student_model.count_params()  + convparams,
            "accuracy_student_premodel_prediction": accuracy_student
        }
        
        student_model = 0 
        lastconv_out = 0
        logit_out = 0 
        test_lastconv_out = 0
        test_logit_out = 0 
        
        results.append(out)

    print('finetunedense2 = ',results)
    plot_findings2(results,'dense2',save=True)


# finetunedense1_dense2()
# remove_d1_d2()
finetunedense2()