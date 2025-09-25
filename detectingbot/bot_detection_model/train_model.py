from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import warnings
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from fileinput import filename
from pyexpat import model
from django.conf import settings
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('TkAgg')
mpl.rcParams['patch.force_edgecolor'] = True
warnings.filterwarnings("ignore")
os.environ['TK_SILENCE_DEPRECATION'] = '1'

class TrainModel():
    def __init__(self):
        self.base_dir = settings.BASE_DIR
        self.labelencoder_X = LabelEncoder()
        self.sc_X = StandardScaler()
        self.classifier = RandomForestClassifier()

    # Importing dataset

    def load_data(self,datafile):
        df = pd.read_csv(
            self.base_dir / 'bot_detection_model' / 'datasets' / datafile)
        return df

    def split_data(self):
        # Splitting data into train and test sets
        df = self.load_data('data.csv')

        # Creating Matrix of Dependent Variable
        X = df.iloc[:, :-1].values

        # Dependent Variable Vector
        Y = df.iloc[:, -1].values

        # Splitting dataset into Trainig set and Test set
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, train_size=0.8, random_state=0)

        return X_train, X_test, Y_train, Y_test

    def label_encode(self, X_train, X_test):
        # Encoding Categorical data

        for i in range(0, 19):
            X_train[:, i] = self.labelencoder_X.fit_transform(
                X_train[:, i].astype(str))
            X_test[:, i] = self.labelencoder_X.fit_transform(
                X_test[:, i].astype(str))

        # Feature Scaling
        X_train = self.sc_X.fit_transform(X_train)
        X_test = self.sc_X.transform(X_test)
        return X_train, X_test

    def train(self):
        print('training started')
        X_train, X_test, Y_train, Y_test = self.split_data()
        X_train, X_test = self.label_encode(X_train, X_test)

        # Fitting the Random Forest Classifier to the Training set
        self.classifier.fit(X_train, Y_train)
        Y_pred = self.classifier.predict(X_test)
        print(f'Y_pred: {Y_pred}')
        
        # Creating the Confusion Matrix
        cm = confusion_matrix(Y_test, Y_pred)

        print(confusion_matrix(Y_test, Y_pred))
        print(classification_report(Y_test, Y_pred))
        print("random forest Accuracy: {0}".format(accuracy_score(Y_test, Y_pred)))

        # make pickle file of our model
        pickle.dump(self.classifier, open(self.base_dir / 'bot_detection_model' / 'outputs' / 'model.pkl', 'wb'))

        # graph
        sns.set(font_scale=1.5)
        sns.set_style("whitegrid", {'axes.grid': False})

        scores_train = self.classifier.predict_proba(X_train)
        scores_test = self.classifier.predict_proba(X_test)

        y_scores_train = []
        y_scores_test = []
        for i in range(len(scores_train)):
            y_scores_train.append(scores_train[i][1])

        for i in range(len(scores_test)):
            y_scores_test.append(scores_test[i][1])

        fpr_dt_train, tpr_dt_train, _ = roc_curve(Y_train, y_scores_train, pos_label=1)
        fpr_dt_test, tpr_dt_test, _ = roc_curve(Y_test, y_scores_test, pos_label=1)

        plt.plot(fpr_dt_train, tpr_dt_train, color='black',
                label='Train AUC: %2f' % auc(fpr_dt_train, tpr_dt_train))
        plt.plot(fpr_dt_test, tpr_dt_test, color='red', ls='--',
                label='Test AUC: %2f' % auc(fpr_dt_test, tpr_dt_test))
        plt.title("RandomForest ROC Curve")
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.legend(loc='lower right')
        plt.savefig(self.base_dir / 'bot_detection_model' / 'outputs' / 'roc_curve.png')
        #plt.show()
        print('training completed')

    def my_predict(self):
        # load the model from disk
        saved_model = pickle.load(open(self.base_dir / 'bot_detection_model' / 'outputs' / 'model.pkl', 'rb'))
        X_test = self.make_predict_data()
        # Predicting the test set result
        Y_pred = saved_model.predict(X_test)
        return Y_pred

    def make_predict_data(self):
        df=self.load_data('test.csv')
        X_test = df.iloc[:, :].values
        for i in range(0, 19):
            X_test[:, i] = self.labelencoder_X.fit_transform(
                X_test[:, i].astype(str))
        X_test = self.sc_X.fit_transform(X_test)
        return X_test
