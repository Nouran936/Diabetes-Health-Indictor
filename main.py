import np as np
import pandas as pd
import os
import pickle
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, FunctionTransformer
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt

#read excel file
dataFile = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")
print('The size of dataset: ', dataFile.shape)
print('------------------------------------------------------------------------------------------------')


#returns boolean value (true or false) whether the data has nulls(true) or not (false)
print('Checking null values: ', dataFile.isnull().values.any())
print('------------------------------------------------------------------------------------------------')


#X is data and y is target
print('-------------------------------X is data and y is target---------------------------')
y = dataFile.iloc[:, 0]
X = dataFile.iloc[:, 1:22]
print('X is \n', X)
print('------------------------------------------------------------------------------------------------')
print('y is \n', y)
print('------------------------------------------------------------------------------------------------')
print('The size of X: ', X.shape)
print('------------------------------------------------------------------------------------------------')
print('The size of y: ', y.shape)
print('------------------------------------------------------------------------------------------------')


#correlation
hm = sns.heatmap(dataFile.corr(), annot=True)
plt.show()


'''#correlation
plt.matshow(dataFile.corr())
plt.show()'''


#oversampling
print('y count before oversampling: \n', y.value_counts())
print('------------------------------------------------------------------------------------------------')
over_sample = SMOTE()
X, y = over_sample.fit_resample(X, y)
print('y count after oversampling: \n', y.value_counts())
print('------------------------------------------------------------------------------------------------')


#Using feature selection by precentile
sel = SelectPercentile(score_func=chi2, percentile=80)
sel.fit(X, y)
X = sel.transform(X)


# -----------------------SCALING-------------------------------

#scaling by standardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

'''#scaling the data by FunctionTransformer
def function1(z):
   return np.sqrt(z)
f = FunctionTransformer(func=function1)
X = f.fit_transform(X)'''

'''#scaling by MinMax
sc = MinMaxScaler(copy=True, feature_range=(0, 1))
X = sc.fit_transform(X)'''

'''nor = Normalizer()
X = nor.fit_transform(X)'''



#Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)


#Model evaluation
def Modl_evaluation(y_test, prediction):
   Acc_score = accuracy_score(y_test, prediction)
   print("Accuracy = ", Acc_score)
   print('------------------------------------------------------------------------------------------------')

   Prec_score = precision_score(y_test, prediction, average='micro')
   print("Precision = ", Prec_score)
   print('------------------------------------------------------------------------------------------------')

   F1_score = f1_score(y_test, prediction, average='micro')
   print("F1 Score = ", F1_score)
   print('------------------------------------------------------------------------------------------------')

   print('classification \n', classification_report(y_test, prediction))

   Conf_Matx = confusion_matrix(y_test, prediction)
   print('Confusion Matrix\n', confusion_matrix(y_test, prediction))
   sns.heatmap(Conf_Matx, center=True)
   plt.show()


#logistic Regression Model
print('------------------------------------------logistic Regression Model---------------------------------------------------')
def log_model():
   filename = 'log_model.sav'
   if os.path.exists('log_model.sav'):
      print("Loading Trained Model")
      # load the model from disk
      loaded_model = pickle.load(open(filename, 'rb'))
      y_pred = loaded_model.predict(X_test)
      Modl_evaluation(y_test, y_pred)
   else:
      print("Creating and training a new model")
      print("Model training started")
      # save the model
      lr = LogisticRegression(max_iter=1000, penalty='none', solver='lbfgs', random_state=33, C=1)
      lr.fit(X_train, y_train)
      pickle.dump(lr, open(filename, 'wb'))
      # load the model from disk
      loaded_model = pickle.load(open(filename, 'rb'))
      y_pred = lr.predict(X_test)
      print('Numbers of iterations: ', lr.n_iter_)
      print('------------------------------------------------------------------------------------------------')
      Modl_evaluation(y_test, y_pred)
log_model()


#Decision tree Model
print('------------------------------------------------Decision Tree----------------------------------------------------------')
def tree_model():
   filename = 'tree_model.sav'
   if os.path.exists('tree_model.sav'):
      print("Loading Trained Model")
      # load the model from disk
      loaded_model = pickle.load(open(filename, 'rb'))
      y_pred = loaded_model.predict(X_test)
      Modl_evaluation(y_test, y_pred)
   else:
      print("Creating and training a new model")
      print("Model training started")
      # save the model
      dt = DecisionTreeClassifier(min_samples_split=30, splitter="best", max_depth=100, random_state=33)
      dt.fit(X_train, y_train)
      pickle.dump(dt, open(filename, 'wb'))
      # load the model from disk
      loaded_model = pickle.load(open(filename, 'rb'))
      y_pred = dt.predict(X_test)
      Modl_evaluation(y_test, y_pred)
tree_model()


#SVM Model
print(('---------------------------------------------------- SVM --------------------------------------------------------------'))
def svm_model():
   filename = 'svm_model.sav'
   if os.path.exists('svm_model.sav'):
      print("Loading Trained Model")
      # load the model from disk
      loaded_model = pickle.load(open(filename, 'rb'))
      y_pred = loaded_model.predict(X_test)
      Modl_evaluation(y_test, y_pred)
   else:
      print("Creating and training a new model")
      print("Model training started")
      # save the model
      classifier = LinearSVC(dual=False, penalty='l2', random_state=33)
      classifier.fit(X_train, y_train)
      pickle.dump(classifier, open(filename, 'wb'))
      # load the model from disk
      loaded_model = pickle.load(open(filename, 'rb'))
      y_pred = loaded_model.predict(X_test)
      Modl_evaluation(y_test, y_pred)
svm_model()




'''plt.boxplot(dataFile)
plt.grid(False)
plt.show()
for i in range(1, 22):
  lowerLimit = dataFile.iloc[:, i].quantile(0.25)
  dataFile[dataFile.iloc[:, i] < lowerLimit]
  upperLimit = dataFile.iloc[:, i].quantile(0.75)
  dataFile[dataFile.iloc[:, i] > upperLimit]
  dataFile = dataFile[(dataFile.iloc[:, i] >= lowerLimit) & (dataFile.iloc[:, i] <= upperLimit)]
  print(dataFile.shape)
plt.boxplot(dataFile.iloc[:, 1:22])
plt.grid(False)
plt.show()'''


