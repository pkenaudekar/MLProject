import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("../MLProject/average_state_rainfall_using_districts_data.csv",sep=",")
data = data.fillna(data.mean())
data.info()

print(data.head())

print(data.describe())

data.hist(figsize=(24,24))

data.groupby("Year").sum().plot(figsize=(12,8))

data[['Year', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']].groupby("Year").sum().plot(figsize=(13,8))

data[['State/UT', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']].groupby("State/UT").mean().plot.barh(stacked=True,figsize=(13,8))

plt.figure(figsize=(11,4))
sns.heatmap(data[['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December', 'Annual']].corr(),annot=True)
plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------
#Function to plot the graphs
def plot_graphs(groundtruth,prediction,title):        
    N = 9
    ind = np.arange(N)  # the x locations for the groups
    width = 0.27       # the width of the bars

    fig = plt.figure()
    fig.suptitle(title, fontsize=12)
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind, groundtruth, width, color='r')
    rects2 = ax.bar(ind+width, prediction, width, color='g')

    ax.set_ylabel("Amount of rainfall")
    ax.set_xticks(ind+width)
    ax.set_xticklabels( ('April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December') )
    ax.legend( (rects1[0], rects2[0]), ('Ground truth', 'Prediction') )

#autolabel(rects1)
    for rect in rects1:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                ha='center', va='bottom')
    for rect in rects2:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                ha='center', va='bottom')
#     autolabel(rects2)

    plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------
# seperation of training and testing data
print("################################ TRAINING ON COMPLETE DATASET ################################")
print()
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

division_data = np.asarray(data[['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']])

X = None; y = None
for i in range(division_data.shape[1]-3):
    if X is None:
        X = division_data[:, i:i+3]
        y = division_data[:, i+3]
    else:
        X = np.concatenate((X, division_data[:, i:i+3]), axis=0)
        y = np.concatenate((y, division_data[:, i+3]), axis=0)
        
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


#-------------------------------------------------------------------------------------------------------------------------------------------
#test 2017
temp = data[['State/UT','January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']].loc[data['Year'] == 2017]

data_2017 = np.asarray(temp[['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']].loc[temp['State/UT'] == 'Uttar Pradesh'])

X_year_2017 = None; y_year_2017 = None
for i in range(data_2017.shape[1]-3):
    if X_year_2017 is None:
        X_year_2017 = data_2017[:, i:i+3]
        y_year_2017 = data_2017[:, i+3]
    else:
        X_year_2017 = np.concatenate((X_year_2017, data_2017[:, i:i+3]), axis=0)
        y_year_2017 = np.concatenate((y_year_2017, data_2017[:, i+3]), axis=0)
#-------------------------------------------------------------------------------------------------------------------------------------------		
#test 2013
temp = data[['State/UT','January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']].loc[data['Year'] == 2013]

data_2013 = np.asarray(temp[['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']].loc[temp['State/UT'] == 'Uttar Pradesh'])

X_year_2013 = None; y_year_2013 = None
for i in range(data_2013.shape[1]-3):
    if X_year_2013 is None:
        X_year_2013 = data_2013[:, i:i+3]
        y_year_2013 = data_2013[:, i+3]
    else:
        X_year_2013 = np.concatenate((X_year_2013, data_2013[:, i:i+3]), axis=0)
        y_year_2013 = np.concatenate((y_year_2013, data_2013[:, i+3]), axis=0)
#-------------------------------------------------------------------------------------------------------------------------------------------	
#test 2009
temp = data[['State/UT','January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']].loc[data['Year'] == 2009]

data_2009 = np.asarray(temp[['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']].loc[temp['State/UT'] == 'Uttar Pradesh'])

X_year_2009 = None; y_year_2009 = None
for i in range(data_2009.shape[1]-3):
    if X_year_2009 is None:
        X_year_2009 = data_2009[:, i:i+3]
        y_year_2009 = data_2009[:, i+3]
    else:
        X_year_2009 = np.concatenate((X_year_2009, data_2009[:, i:i+3]), axis=0)
        y_year_2009 = np.concatenate((y_year_2009, data_2009[:, i+3]), axis=0)
#-------------------------------------------------------------------------------------------------------------------------------------------		
from sklearn import linear_model

# linear model
#Linear regression with combined L1 and L2 priors as regularizer.
#alpha : float, optional
#Constant that multiplies the penalty terms. Defaults to 1.0. See the notes for the 
#exact mathematical meaning of this parameter.``alpha = 0`` is equivalent to an ordinary 
#least square, solved by the LinearRegression object. For numerical reasons, using alpha = 0 
#with the Lasso object is not advised. Given this, you should use the LinearRegression object.

#ElasticNet is hybrid of Lasso and Ridge Regression techniques. It is trained
#with L1 and L2 prior as regularizer. Elastic-net is useful when there are
#multiple features which are correlated. Lasso is likely to pick one of these
#at random, while elastic-net is likely to pick both. 
reg = linear_model.ElasticNet(alpha=0.5)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print("*************** RESULTS USING LINEAR MODEL ***************")
print()
print ("MEAN ABSOLUTE ERROR")
print (mean_absolute_error(y_test, y_pred))
#-------------------------------------------------------------------------------------------------------------------------------------------

#2017
y_year_pred_2017 = reg.predict(X_year_2017)

#2013
y_year_pred_2013 = reg.predict(X_year_2013)
   
#2009
y_year_pred_2009 = reg.predict(X_year_2009)

print ("MEAN 2017")
print (np.mean(y_year_2017),np.mean(y_year_pred_2017))
print ("Standard deviation 2017")
print (np.sqrt(np.var(y_year_2017)),np.sqrt(np.var(y_year_pred_2017)))


print ("MEAN 2013")
print (np.mean(y_year_2013),np.mean(y_year_pred_2013))
print ("Standard deviation 2013")
print (np.sqrt(np.var(y_year_2013)),np.sqrt(np.var(y_year_pred_2013)))


print ("MEAN 2009")
print (np.mean(y_year_2009),np.mean(y_year_pred_2009))
print ("Standard deviation 2009")
print (np.sqrt(np.var(y_year_2009)),np.sqrt(np.var(y_year_pred_2009)))


plot_graphs(y_year_2017,y_year_pred_2017,"Year-2017")
plot_graphs(y_year_2013,y_year_pred_2013,"Year-2013")
plot_graphs(y_year_2009,y_year_pred_2009,"Year-2009")

#-------------------------------------------------------------------------------------------------------------------------------------------

from sklearn.svm import SVR

# SVM model
#Epsilon-Support Vector Regression.
#gamma : float, optional (default=’auto’)
#Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.Current default is ‘auto’ which uses 1 / n_features.
#C : float, optional (default=1.0)
#Penalty parameter C of the error term.
#epsilon : float, optional (default=0.1)
#Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty is 
#associated in the training loss function with points predicted within a distance epsilon from the actual value.

clf = SVR(gamma='auto', C=0.1, epsilon=0.2)
clf.fit(X_train, y_train) 
y_pred = clf.predict(X_test)
print()
print("*************** RESULTS USING SVR MODEL ***************")
print()
print ("MEAN ABSOLUTE ERROR")
print (mean_absolute_error(y_test, y_pred))
#-------------------------------------------------------------------------------------------------------------------------------------------
#2017
y_year_pred_2017 = reg.predict(X_year_2017)

#2013
y_year_pred_2013 = reg.predict(X_year_2013)
    
#2009
y_year_pred_2009 = reg.predict(X_year_2009)
print ("MEAN 2009")
print (np.mean(y_year_2009),np.mean(y_year_pred_2009))
print ("Standard deviation 2009")
print (np.sqrt(np.var(y_year_2009)),np.sqrt(np.var(y_year_pred_2009)))


print ("MEAN 2013")
print (np.mean(y_year_2013),np.mean(y_year_pred_2013))
print ("Standard deviation 2013")
print (np.sqrt(np.var(y_year_2013)),np.sqrt(np.var(y_year_pred_2013)))


print ("MEAN 2017")
print (np.mean(y_year_2017),np.mean(y_year_pred_2017))
print ("Standard deviation 2017")
print (np.sqrt(np.var(y_year_2017)),np.sqrt(np.var(y_year_pred_2017)))

plot_graphs(y_year_2009,y_year_pred_2009,"Year-2009")
plot_graphs(y_year_2013,y_year_pred_2013,"Year-2013")
plot_graphs(y_year_2017,y_year_pred_2017,"Year-2017")

#-------------------------------------------------------------------------------------------------------------------------------------------

from keras.models import Model
from keras.layers import Dense, Input, Conv1D, Flatten
# NN model

# Input tensor for sequences of 3 timesteps,
# each containing a 1-dimensional vector
inputs = Input(shape=(3,1))

#1D convolution layer 
#filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
#kernel_size: An integer or tuple/list of a single integer, specifying the length of the 1D convolution window.
#padding: One of "valid", "causal" or "same" (case-insensitive). "valid" means "no padding". "same" results 
#in padding the input such that the output has the same length as the original input. "causal" results in causal 
#(dilated) convolutions, e.g. output[t] does not depend on input[t + 1:]. A zero padding is used such that the 
#output has the same length as the original input. Useful when modeling temporal data where the model should not violate the temporal order. 
#activation: Activation function to use . If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x),
#elu-exponential linear unit.ELU is a function that tend to converge cost to zero faster and produce more accurate results
x = Conv1D(64, 2, padding='same', activation='elu')(inputs)
x = Conv1D(128, 2, padding='same', activation='elu')(x)

#Flattens the input
x = Flatten()(x)

#Dense implements the operation: output = activation(dot(input, kernel) + bias) where activation is the element-wise activation 
#function passed as the activation argument, kernel is a weights matrix created by the layer, and bias is a bias vector created 
#by the layer (only applicable if use_bias is True).
#keras.layers.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
# kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
x = Dense(128, activation='elu')(x)
x = Dense(64, activation='elu')(x)
x = Dense(32, activation='elu')(x)
x = Dense(1, activation='linear')(x)

#Model will include all layers required in the computation of [inputs] given [x]
model = Model(inputs=[inputs], outputs=[x])

#Configures the model for training
#optimizer: String (name of optimizer) or optimizer instance.
#loss: String (name of objective function) or objective function or Loss instance.
#metrics: List of metrics to be evaluated by the model during training and testing.
#Typically you will use metrics=['accuracy']. To specify different metrics for different 
#outputs of a multi-output model, you could also pass a dictionary, such as metrics={'output_a': 'accuracy', 'output_b': ['accuracy', 'mse']}. 
#You can also pass a list (len = len(outputs)) of lists of metrics such as metrics=[['accuracy'], ['accuracy', 'mse']] or 
#metrics=['accuracy', ['accuracy', 'mse']].
model.compile(loss='mean_squared_error', optimizer='adamax', metrics=['mae'])
model.summary()
#-------------------------------------------------------------------------------------------------------------------------------------------

#Trains the model for a fixed number of epochs (iterations on a dataset).
#x: Input data, y: Target data, 
#batch_size: Integer or None. Number of samples per gradient update. 
#If unspecified, batch_size will default to 32. Do not specify the batch_size if your data is in the 
#form of symbolic tensors, generators, or Sequence instances (since they generate batches).
#epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.
# Note that in conjunction with initial_epoch, epochs is to be understood as "final epoch". The model is not trained for
# a number of iterations given by epochs, but merely until the epoch of index epochs is reached.
#verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch
#validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart 
#this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch.
#shuffle: Boolean (whether to shuffle the training data before each epoch) or str (for 'batch'). 'batch' is a special option for dealing with the 
#limitations of HDF5 data; it shuffles in batch-sized chunks. Has no effect when steps_per_epoch is not None.
model.fit(x=np.expand_dims(X_train, axis=2), y=y_train, batch_size=64, epochs=10, verbose=1, validation_split=0.1, shuffle=True)
y_pred = model.predict(np.expand_dims(X_test, axis=2))
print()
print("*************** RESULTS USING NEURAL NETWORK MODEL ***************")
print()
print ("MEAN ABSOLUTE ERROR")
print (mean_absolute_error(y_test, y_pred))

#-------------------------------------------------------------------------------------------------------------------------------------------

#2009
y_year_pred_2009 = reg.predict(X_year_2009)

#2013
y_year_pred_2013 = reg.predict(X_year_2013)
    
#2017
y_year_pred_2017 = reg.predict(X_year_2017)

print ("MEAN 2009")
print (np.mean(y_year_2009),np.mean(y_year_pred_2009))
print ("Standard deviation 2009")
print (np.sqrt(np.var(y_year_2009)),np.sqrt(np.var(y_year_pred_2009)))


print ("MEAN 2013")
print (np.mean(y_year_2013),np.mean(y_year_pred_2013))
print ("Standard deviation 2013")
print (np.sqrt(np.var(y_year_2013)),np.sqrt(np.var(y_year_pred_2013)))


print ("MEAN 2017")
print (np.mean(y_year_2017),np.mean(y_year_pred_2017))
print ("Standard deviation 2017")
print (np.sqrt(np.var(y_year_2017)),np.sqrt(np.var(y_year_pred_2017)))

plot_graphs(y_year_2009,y_year_pred_2009,"Year-2009")
plot_graphs(y_year_2013,y_year_pred_2013,"Year-2013")
plot_graphs(y_year_2017,y_year_pred_2017,"Year-2017")

#-------------------------------------------------------------------------------------------------------------------------------------------

# spliting training and testing data only for Uttar Pradesh
print()
print("################################ TRAINING ON UTTAR PRADESH DATASET ################################")
uttarpradesh = np.asarray(data[['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']].loc[data['State/UT'] == 'Uttar Pradesh'])

X = None; y = None
for i in range(uttarpradesh.shape[1]-3):
    if X is None:
        X = uttarpradesh[:, i:i+3]
        y = uttarpradesh[:, i+3]
    else:
        X = np.concatenate((X, uttarpradesh[:, i:i+3]), axis=0)
        y = np.concatenate((y, uttarpradesh[:, i+3]), axis=0)
        
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

#-------------------------------------------------------------------------------------------------------------------------------------------


from sklearn import linear_model

# linear model
reg = linear_model.ElasticNet(alpha=0.5)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print()
print("*************** RESULTS USING LINEAR MODEL ***************")
print()
print ("MEAN ABSOLUTE ERROR")
print (mean_absolute_error(y_test, y_pred))

#-------------------------------------------------------------------------------------------------------------------------------------------

#2017
y_year_pred_2017 = reg.predict(X_year_2017)

#2013
y_year_pred_2013 = reg.predict(X_year_2013)
   
#2009
y_year_pred_2009 = reg.predict(X_year_2009)

print ("MEAN 2017")
print (np.mean(y_year_2017),np.mean(y_year_pred_2017))
print ("Standard deviation 2017")
print (np.sqrt(np.var(y_year_2017)),np.sqrt(np.var(y_year_pred_2017)))


print ("MEAN 2013")
print (np.mean(y_year_2013),np.mean(y_year_pred_2013))
print ("Standard deviation 2013")
print (np.sqrt(np.var(y_year_2013)),np.sqrt(np.var(y_year_pred_2013)))


print ("MEAN 2009")
print (np.mean(y_year_2009),np.mean(y_year_pred_2009))
print ("Standard deviation 2009")
print (np.sqrt(np.var(y_year_2009)),np.sqrt(np.var(y_year_pred_2009)))


plot_graphs(y_year_2017,y_year_pred_2017,"Year-2017")
plot_graphs(y_year_2013,y_year_pred_2013,"Year-2013")
plot_graphs(y_year_2009,y_year_pred_2009,"Year-2009")


#-------------------------------------------------------------------------------------------------------------------------------------------

from sklearn.svm import SVR

# SVM model
clf = SVR(kernel='rbf', gamma='auto', C=0.5, epsilon=0.2)
clf.fit(X_train, y_train) 
y_pred = clf.predict(X_test)
print()
print("*************** RESULTS USING SVR MODEL ***************")
print()
print ("MEAN ABSOLUTE ERROR")
print (mean_absolute_error(y_test, y_pred))

#-------------------------------------------------------------------------------------------------------------------------------------------

#2017
y_year_pred_2017 = reg.predict(X_year_2017)

#2013
y_year_pred_2013 = reg.predict(X_year_2013)
    
#2009
y_year_pred_2009 = reg.predict(X_year_2009)
print ("MEAN 2009")
print (np.mean(y_year_2009),np.mean(y_year_pred_2009))
print ("Standard deviation 2009")
print (np.sqrt(np.var(y_year_2009)),np.sqrt(np.var(y_year_pred_2009)))


print ("MEAN 2013")
print (np.mean(y_year_2013),np.mean(y_year_pred_2013))
print ("Standard deviation 2013")
print (np.sqrt(np.var(y_year_2013)),np.sqrt(np.var(y_year_pred_2013)))


print ("MEAN 2017")
print (np.mean(y_year_2017),np.mean(y_year_pred_2017))
print ("Standard deviation 2017")
print (np.sqrt(np.var(y_year_2017)),np.sqrt(np.var(y_year_pred_2017)))

plot_graphs(y_year_2009,y_year_pred_2009,"Year-2009")
plot_graphs(y_year_2013,y_year_pred_2013,"Year-2013")
plot_graphs(y_year_2017,y_year_pred_2017,"Year-2017")

#-------------------------------------------------------------------------------------------------------------------------------------------

from keras.models import Model
from keras.layers import Dense, Input, Conv1D, Flatten
# NN model
inputs = Input(shape=(3,1))
x = Conv1D(64, 2, padding='same', activation='elu')(inputs)
x = Conv1D(128, 2, padding='same', activation='elu')(x)
x = Flatten()(x)
x = Dense(128, activation='elu')(x)
x = Dense(64, activation='elu')(x)
x = Dense(32, activation='elu')(x)
x = Dense(1, activation='linear')(x)
model = Model(inputs=[inputs], outputs=[x])
model.compile(loss='mean_squared_error', optimizer='adamax', metrics=['mae'])
model.summary()
#-------------------------------------------------------------------------------------------------------------------------------------------
model.fit(x=np.expand_dims(X_train, axis=2), y=y_train, batch_size=64, epochs=10, verbose=1, validation_split=0.1, shuffle=True)
y_pred = model.predict(np.expand_dims(X_test, axis=2))
print()
print("*************** RESULTS USING NEURAL NETWORK MODEL ***************")
print()
print ("MEAN ABSOLUTE ERROR")
print (mean_absolute_error(y_test, y_pred))

#-------------------------------------------------------------------------------------------------------------------------------------------
#2009
y_year_pred_2009 = reg.predict(X_year_2009)

#2013
y_year_pred_2013 = reg.predict(X_year_2013)
    
#2017
y_year_pred_2017 = reg.predict(X_year_2017)

print ("MEAN 2009")
print (np.mean(y_year_2009),np.mean(y_year_pred_2009))
print ("Standard deviation 2009")
print (np.sqrt(np.var(y_year_2009)),np.sqrt(np.var(y_year_pred_2009)))


print ("MEAN 2013")
print (np.mean(y_year_2013),np.mean(y_year_pred_2013))
print ("Standard deviation 2013")
print (np.sqrt(np.var(y_year_2013)),np.sqrt(np.var(y_year_pred_2013)))


print ("MEAN 2017")
print (np.mean(y_year_2017),np.mean(y_year_pred_2017))
print ("Standard deviation 2017")
print (np.sqrt(np.var(y_year_2017)),np.sqrt(np.var(y_year_pred_2017)))

plot_graphs(y_year_2009,y_year_pred_2009,"Year-2009")
plot_graphs(y_year_2013,y_year_pred_2013,"Year-2013")
plot_graphs(y_year_2017,y_year_pred_2017,"Year-2017")


