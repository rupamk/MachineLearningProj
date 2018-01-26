#------models------
import pandas
import keras
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.optimizers import SGD, Adadelta, Adagrad, Nadam,Adamax
import numpy as np
import os
import tensorflow as tf
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.externals import joblib
from tensorflow.contrib import learn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt

class models():
    def create_model(self, input_shape, optimizer,dropout,neurons,num_layers,initialize = 'uniform', activation='relu',loss='binary_crossentropy'):
        num_classes=1
        # create model
        dnn_model = Sequential()
        #layer1
        dnn_model.add(Dense(neurons, kernel_initializer=initialize,activation=activation, input_shape=(input_shape,)))
        #dnn_model.add(Dropout(dropout))

        for i in range(1,num_layers):
            dnn_model.add(Dense(neurons, kernel_initializer=initialize, activation=activation))
            dnn_model.add(Dropout(dropout))

        dnn_model.add(Dense(num_classes, kernel_initializer=initialize, activation='sigmoid'))
        # Compile model
        dnn_model.compile(loss=loss,
                          optimizer=optimizer,
                          metrics=['accuracy'])
        dnn_model.summary()
        return dnn_model

    def __init__(self, path):
        data_df = pandas.read_csv(path + '/Training_data.csv', header=None)
        x1 = data_df.values.astype('float32')

        label_df = pandas.read_csv(path + '/Training_labels.csv', header=None)
        y1 = label_df.values.astype('int64')

        data_df = pandas.read_csv(path + '/Test_data.csv', header=None)
        x2 = data_df.values.astype('float32')

        label_df = pandas.read_csv(path + '/Test_labels.csv', header=None)
        y2 = label_df.values.astype('int64')



        x_train = x1
        y_train = y1

        # randomize the data
        perm = np.arange(x_train.shape[0])
        np.random.shuffle(perm)
        self.x_train = x_train[perm]
        self.y_train = y_train[perm]

        self.x_test = x2
        self.y_test = y2



    def run_DNN(self,x_train,y_training,x_test,y_testing,sector_id):
        num_classes = 1
        batch_size = 500
        epochs = 10
        loss=[]
        accuracy=[]
        lines = []

        '''
        #select the best possible values
        optimizer = ['Adamax','Adadelta','Adagrad','RMSprop','Adam','SGD']
        RATE = [0.1, 0.001, 0.0001]
        num_layers = [1,2,4]
        neurons = [64,512,1000]
        epochs =[5,10]
        batch_size=[100,500]

        '''
        ran = np.linspace(0,11,12)
        for i in ran:
            i = int(i)
            y_train = y_training[:, i]

            y_test = y_testing[:, i]
            
            
            #validation_data=(x_test,y_test)
            #best optimizer: Optimizer: Adam Learning rate: 0.001|
            opt =Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
            model = self.create_model(x_train.shape[1], opt, 0.2, 500,5, 'uniform' , 'relu', 'binary_crossentropy')
            history=model.fit(x_train, y_train,validation_data=(x_test,y_test), batch_size=batch_size,
                      epochs=epochs,
                      verbose=1)

            score = model.evaluate(x_test, y_test, verbose=1)

            print('Test loss:', score[0])
            print('Test accuracy:', score[1])
            loss.append(score[0])
            accuracy.append(score[1])

            lines += plt.plot(history.history['acc'], label="Sector:{} Train".format(i) )
            lines +=plt.plot(history.history['val_acc'], label="Sector:{} Test".format(i))

        '''
        y_train = y_training[:, sector_id]

        y_test = y_testing[:, sector_id]
        # to select the best optimizer

        # optimizers
        nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        ada_max = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        ada_delta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
        ada_grad = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
        adm = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
        rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        sgd = SGD(lr=0.01, decay=0.99, momentum=0.9, nesterov=True)

        # model = KerasClassifier(build_fn=self.create_model, verbose=0)
        # create_model(self, input_shape, optimizer,dropout,neurons, activation='relu',loss='binary_crossentropy'):
        for n in neurons:
            for layers in num_layers:
                for j in optimizer:
                    for batch in batch_size:
                        for epoch in epochs:
                            for k in range(len(RATE)):
                                if j == 'Nadam':
                                    opt = Nadam(lr=RATE[k], beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                                                schedule_decay=0.004)
                                elif j == 'Adamax':
                                    opt = Adamax(lr=RATE[k], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
                                elif j == 'Adadelta':
                                    opt = Adadelta(lr=RATE[k], rho=0.95, epsilon=1e-08, decay=0.0)
                                elif j == 'Adagrad':
                                    opt = Adagrad(lr=RATE[k], epsilon=1e-08, decay=0.0)
                                elif j == 'Adam':
                                    opt = Adam(lr=RATE[k], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
                                elif j == 'RMSprop':
                                    opt = RMSprop(lr=RATE[k], rho=0.9, epsilon=1e-08, decay=0.0)
                                else:
                                    opt = SGD(lr=RATE[k], decay=0.99, momentum=0.9, nesterov=True)
                                print(
                                    "|-------------------------------------------------------------------------------|")
                                print(
                                    "|Optimizer:" + str(j) + " Learning rate:" + str(RATE[k]) + " Num_Neurons:" + str(
                                        n) + " Num_Layers: " + str(layers+2) + " Batch_size:" + str(batch)+" Epochs:"+str(epoch)+"|")
                                print(
                                    "|-------------------------------------------------------------------------------|")
                                model = self.create_model(x_train.shape[1], opt, 0.2, n, layers, 'uniform', 'relu',
                                                          'binary_crossentropy')
                                model.fit(x_train, y_train, batch_size=batch,
                                          epochs=epoch,
                                          verbose=0)

                                score = model.evaluate(x_test, y_test, verbose=0)

                                print('Test loss:', score[0])
                                print('Test accuracy:', score[1])
        '''
        print("|-------------------------------------------------------------------------------|")
        print("|-----------------------------------DNN-----------------------------------------|")
        print("|-------------------------------------------------------------------------------|")
        print()
        print("Accuracy:")
        print(accuracy)
        print("The mean of accuracy is: " + str(np.mean(accuracy)))
        print("Loss")
        print(loss)

        # summarize history for accuracy
        '''
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        labels = [l.get_label() for l in lines]
        plt.legend(lines, labels, loc='upper left')
        plt.show()
        '''

    
    def run_Linear_SVM(self,x_training,y_training,x_testing,y_testing):
        print("|-------------------------------------------------------------------------------|")
        print("|---------------------------------Linear SVM------------------------------------|")
        print("|-------------------------------------------------------------------------------|")
        print()

        y_train = y_training[:, 0]

        y_test = y_testing[:, 0]

        n_classes = len(set(y_train))

        Liner_SVM = learn.LinearClassifier(
            feature_columns=[tf.contrib.layers.real_valued_column("", dimension=x_training.shape[1])],
            n_classes=n_classes, optimizer=tf.train.FtrlOptimizer(
            learning_rate=0.1,
            l1_regularization_strength=0.001
        ))
        
        # Set the parameters by cross-validation
        learning_rate = [0.1]
        l1_regularization_strength= [0.1]                  

        best_rate = 0
        best_reg =0
        best_accuracy =100;
        scores = ['accuracy']  # metric for testing
        print("# Tuning hyper-parameters for %s" % scores[0])
        for rate in learning_rate:
            for reg in l1_regularization_strength:   
                print()
                clf = learn.LinearClassifier(
            feature_columns=[tf.contrib.layers.real_valued_column("", dimension=x_training.shape[1])],
            n_classes=n_classes, optimizer=tf.train.FtrlOptimizer(
            learning_rate=rate,
            l1_regularization_strength=reg
        ))
                # fit model
                clf.fit(x_training, y_train)
                y_pred = list(Liner_SVM.predict(x_testing))
                acc = sklearn.metrics.accuracy_score(y_test, y_pred)
                if best_accuracy>acc:
                    best_accuracy = acc
                    best_rate = rate
                    best_reg = reg

                print('The accuracy obtained for learning rate:' + str(rate)+ ' l1_regularization_strength:' + str(ref)+' is:'+str(best_accuracy))

            
     
        Liner_SVM = learn.LinearClassifier(
            feature_columns=[tf.contrib.layers.real_valued_column("", dimension=x_training.shape[1])],
            n_classes=n_classes, optimizer=tf.train.FtrlOptimizer(
            learning_rate=best_rate,
            l1_regularization_strength=best_reg
        ))
        accuracy_store = []
        print("Accuracy for sector " + str(0) + " : " + str(
            sklearn.metrics.accuracy_score(y_test, y_pred)) + " and % of 1's in Test Data : " + str(y_test.mean()))
        for i in range(1,y_training.shape[1]):
            y_train = y_training[:,i]

            y_test = y_testing[:,i]

            n_classes = len(set(y_train))

            Liner_SVM.fit(x_training, y_train, steps=2000)

            y_pred = list(Liner_SVM.predict(x_testing))

            print("Accuracy for sector " + str(i) + " : " + str(
                sklearn.metrics.accuracy_score(y_test, y_pred)) + " and % of 1's in Test Data : " + str(
                y_test.mean()))
            accuracy_store.append(sklearn.metrics.accuracy_score(y_test, y_pred))
            # Evaluate and report metrics.
            #eval_metrics = classifier.evaluate(input_fn=y_test, steps=1)
            #print(eval_metrics)
        print("The average accuracy is : " + str(np.mean(accuracy_store)))
    

    def run_kernel_SVM(self,x_training,y_training,x_testing,y_testing):
        print("|-------------------------------------------------------------------------------|")
        print("|---------------------------------Kernel SVM------------------------------------|")
        print("|-------------------------------------------------------------------------------|")
        print()

        y_train = y_training[:, 0]
        y_test = y_testing[:, 0]

        # Set the parameters
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-09, 1e-06, 1e-3, 1e0,1e3],
                     'C': [1, 10, 100, 1000]},
                            {'kernel': ['poly'], 'gamma': [1e-09, 1e-06, 1e-3, 1e0,1e3],
                             'C': [1, 10, 100, 1000], 'degree':[3,4,5,6]},{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
                            ]

        scores = ['accuracy']  # metric for testing


        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()
            clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='%s' % score)
            # fit model
            clf.fit(x_training, y_train)

            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores for all parameters:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            print()

        best_params = clf.best_params_
        if best_params['kernel'] == 'poly':
            kernel_SVM = SVC(C=best_params['C'], degree = best_params['degree'], kernel=best_params['kernel'], gamma=best_params['gamma'])
        else:
            kernel_SVM = SVC(C=best_params['C'], kernel = best_params['kernel'], gamma = best_params['gamma'])

        accuracy_store = []


        for i in range(y_training.shape[1]):
            y_train = y_training[:, i]

            y_test = y_testing[:, i]

            kernel_SVM.fit(x_training, y_train)
            y_pred = list(kernel_SVM.predict(x_testing))
            print("Accuracy for sector " + str(i) + " : " + str(
                sklearn.metrics.accuracy_score(y_test, y_pred)) + " and % of 1's in Test Data : " + str(
                y_test.mean()))
            accuracy_store.append(sklearn.metrics.accuracy_score(y_test, y_pred))

        print("The average accuracy is : " + str(np.mean(accuracy_store)))

        '''    
        for i in range(y_training.shape[1]):
           y_train = y_training[:, i]

           y_test = y_testing[:, i]

           n_classes = len(set(y_train))
           optimizer = tf.train.FtrlOptimizer(learning_rate=50.0, l2_regularization_strength=0.001)

           feature_columns = tf.contrib.layers.real_valued_column("", dimension=x_training.shape[1])

           #kernel
           kernel_mapper = tf.contrib.kernel_methods.RandomFourierFeatureMapper(
               input_dim=x_training.shape[1], output_dim=2000, stddev=5.0, name='rffm')
           kernel_mappers = {feature_columns: [kernel_mapper]}

           classifier = tf.contrib.kernel_methods.KernelLinearClassifier(n_classes=n_classes, optimizer=optimizer, kernel_mappers=kernel_mappers)

           classifier.fit(x_training, y_train, steps=2000)

           y_pred = list(classifier.predict(x_testing))

           print("Accuracy for sector: " + str(i))
           print(sklearn.metrics.accuracy_score(y_test, y_pred))
        '''




    def run_Logistic_regression(self, x_training,y_training,x_testing,y_testing,sector_id):
    	
        print("|-------------------------------------------------------------------------------|")
        print("|------------------------------Logistic Regression------------------------------|")
        print("|-------------------------------------------------------------------------------|")
        print()
        
        y_train = y_training[:,sector_id]
        y_test = y_testing[:,sector_id]
        
        # Set the parameters
        tuned_parameters = [
                            {'solver': ['lbfgs'], 'max_iter': [100],
                             'C': [1e-3,1e-2,1e-1,1, 10,100], 'warm_start': ['True','False'], 'penalty':['l2']},
                            {'solver': ['liblinear'], 'max_iter': [100],
                             'C': [1e-3, 1e-2,1e-1,1,10,100], 'penalty':['l2']},
                            {'solver': ['sag'], 'max_iter': [500],
                             'C': [1e-3, 1e-2,1e-1,1, 10, 100], 'warm_start': ['True','False'], 'penalty':['l2']}
                            ]
        scores = ['accuracy'] #metric for testing
        # instantiate model
        logreg = LogisticRegression()

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()
            clf = GridSearchCV(logreg, tuned_parameters,scoring='%s' % score)
            # fit model
            clf.fit(x_training, y_train)
            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores for all parameters:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            print()


        best_params = clf.best_params_

        if best_params['solver'] == 'liblinear':
            logreg = LogisticRegression(C=best_params['C'], class_weight=None, dual=False, fit_intercept=True,
                                        intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                                        penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
                                        verbose=0, warm_start=False)

        else:
            logreg = LogisticRegression(C=100, class_weight=None, dual=False, fit_intercept=True,
                                intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                                penalty='l2', random_state=None, solver=best_params['solver'], tol=0.0001,
                                verbose=0, warm_start=best_params['warm_start'])
                                
        accuracy_store = []


        for i in range(y_training.shape[1]):
            y_train = y_training[:, i]

            y_test = y_testing[:, i]

            logreg.fit(x_training, y_train)

            y_pred = list(logreg.predict(x_testing))
            print("Accuracy for sector " + str(i) + " : " + str(sklearn.metrics.accuracy_score(y_test, y_pred)) + " and % of 1's in Test Data : " + str(y_test.mean()))
            accuracy_store.append(sklearn.metrics.accuracy_score(y_test, y_pred))

        print('Accuracy:')
        print(accuracy_store)
        print("The average accuracy is : " + str(np.mean(accuracy_store)))
