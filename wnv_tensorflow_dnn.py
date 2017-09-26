# This script contains functions used for data loading, cleaning, merging, feature engineering,
# and saving predictions
# It also contains a stacking function, used to obtain meta-features: predicted log(number of mosquitos) for the 2d stage
# Finally, there is a bagging (bootstrap) function designed to stabilize predictions of the Neural Net
# classifier
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import csv
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, asin

from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt

import sklearn.metrics as metrics
import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_io

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn import metrics
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

'''
    Kaggle West Nile virus prediction challenge.

'''

path_train = "train.csv"
path_test= "test2.csv"
path_weather="weather.csv"

def buildLaggedFeatures(df, lag):  # df - dataframe, lag - list of numbers defining lagged values. Builds lagged weather features
    new_dict={}
    for col_name in df:
        new_dict[col_name]=df[col_name]
        # create lagged Series
        for l in lag:
            if col_name!='Date' and col_name!='Station':
                new_dict['%s_lag%d' %(col_name,l)]=df[col_name].shift(l)
    res=pd.DataFrame(new_dict,index=df.index)
    return res

def DuplicatedRows(df): # Calculates number of duplicated rows by Date, Trap, Species
    grouped = df.groupby(['Date', 'Trap', 'Species'])
    num=grouped.count().Latitude.to_dict()
    df['N_Dupl']=-999
    for idx in df.index:
        d = df.loc[idx, 'Date']
        t = df.loc[idx, 'Trap']
        s = df.loc[idx, 'Species']
        df.loc[idx, 'N_Dupl'] = num[(d, t, s)]
    return df

def haversine(lat1, lon1, lat2, lon2): # Calculates the haversine distance between two Lat, Long pairs
    R = 6372.8 # Earth radius in kilometers
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    a = np.power(np.sin(dLat/2), 2) + np.multiply(np.cos(lat1), np.multiply(np.cos(lat2), np.power(np.sin(dLon/2), 2)))
    c = 2*np.arcsin(np.sqrt(a))
    return R * c

def ClosestStation(df): # identify the closest weather station
    df['lat1']=41.995   # latitude of 1st station
    df['lat2']=41.786   # latitude of 2d station
    df['lon1']=-87.933  # longitude of 1st station
    df['lon2']=-87.752  # longitude of 2d station
    df['dist1'] = haversine(df.Latitude.values, df.Longitude.values, df.lat1.values, df.lon1.values) #calculate distance
    df['dist2'] = haversine(df.Latitude.values, df.Longitude.values, df.lat2.values, df.lon2.values)
    indicator = np.less_equal(df.dist1.values, df.dist2.values) # determine which station is the closest
    st = np.ones(df.shape[0])
    st[indicator==0]=2
    df['Station']=st    # obtain station identifier for each row
    df.drop(['dist1', 'dist2', 'lat1', 'lat2', 'lon1', 'lon2' ], axis=1, inplace=True)
    return df

def DateSplit(df):   # parse date in text format to get 3 separate columns for year, month, day
    df = df.copy()
    df['Year'] = pd.DatetimeIndex(df['Date']).year
    df['Month'] = pd.DatetimeIndex(df['Date']).month
    df['Day'] = pd.DatetimeIndex(df['Date']).day
    return df

def GetDummies(df):   # Constructs dummy indicators for species
    dummies=pd.get_dummies(df['Species'])
    df = pd.concat([df, dummies], axis=1)
    return df

def ExtendWnv(df):  # assigns WNV indicator to duplicate rows (if WNV=1, assign 1 to all duplicate rows)
    grouped = df.groupby(by=['Date', 'Trap', 'Species'], as_index=False)['WnvPresent'].max()
    df.drop('WnvPresent', axis=1, inplace=True)
    grouped.columns = ['Date', 'Trap', 'Species', 'WnvPresent']
    result = df.merge(grouped, on=['Date', 'Trap', 'Species'], how="left") #.reset_index()
    return result

def MergeWeather(df1, df2):
    result = df1.merge(df2, on=['Date', 'Station'], how="left",  left_index=True)
    return result

def LoadWeather():  # Load and prepare weather data
    days = [1, 3, 5, 8, 12]     #lagged weather values used as features
    weather = pd.read_csv(path_weather)
    weather.sort_values(['Date', 'Station'], axis=0, ascending=True, inplace=True)
    filter_out = ['Heat', 'CodeSum', 'Depth', 'Water1', 'SnowFall', 'StnPressure',  'SeaLevel', 'AvgSpeed' ]
    weather.drop(filter_out, axis=1, inplace=True) # these variables are not used in the analysis
    weather.replace(['  T','M','-'], [0.001, np.nan, np.nan], inplace=True) # replace "Trace" with 0.001, replace M and missing with NaN
    weather.WetBulb.fillna(method='bfill', inplace=True)  # replace missing WetBulb of 1st station with the value of 2d station
    weather.fillna(method='pad', inplace=True)   # replace all missing values of 2d station with values of 1st station
    weather1 = buildLaggedFeatures(weather[weather['Station']==1], days) # build lagged features for 1st station
    weather2 = buildLaggedFeatures(weather[weather['Station']==2], days) # build lagged features for 2d station
    weather = weather1.append(weather2)                                  # append data from 2 stations
    weather.sort_values(['Date', 'Station'], axis=0, ascending=True, inplace=True)
    return weather

def LoadTest(version): # Load and prepare test data
    print("Loading Test Data")
    test = pd.read_csv(path_test)
    test = DateSplit(test)
    test = DuplicatedRows(test)
    if version == 2: # calculate duplicated rows for various subgroups of data
         l = [['Date', 'Species'], ['Date', 'Trap'], ['Date', 'Block'], ['Year', 'Species'],
              ['Year', 'Trap'], ['Year', 'Block'], ['Month', 'Species'], ['Month', 'Trap'],
              ['Month', 'Block']]
         for i in range(len(l)):
             test = AddMoreFeatures(test, l[i]) # compute number of duplicated rows for each group
    test = ClosestStation(test)
    weather = LoadWeather()
    test = MergeWeather(test, weather)
    test.replace(['UNSPECIFIED CULEX'], ['CULEX PIPIENS'], inplace=True) # replace Unspecified species with PIPIENS
    test = GetDummies(test)
    filter_out = ['Address', 'Block', 'Street', 'Trap', 'AddressNumberAndStreet', 'AddressAccuracy',
                 'Date', 'Species', 'Station' ]
    test.drop(filter_out, axis=1, inplace=True) # these features are not used in prediction
    return test

def LoadTrain(version): # Load and prepare train data
    print("Loading Train Data")
    train = pd.read_csv(path_train)
    train = DateSplit(train)
    train = DuplicatedRows(train)
    if version == 2: # calculate duplicated rows for various subgroups of data
         l = [['Date', 'Species'], ['Date', 'Trap'], ['Date', 'Block'], ['Year', 'Species'],
              ['Year', 'Trap'], ['Year', 'Block'], ['Month', 'Species'], ['Month', 'Trap'],
              ['Month', 'Block']]
         for i in range(len(l)):
             train = AddMoreFeatures(train, l[i]) # compute number of duplicated rows for each group
    train = ExtendWnv(train)
    train = ClosestStation(train)
    weather = LoadWeather()
    train = MergeWeather(train, weather)
    train = GetDummies(train)
    filter_out = ['Address', 'Block', 'Street', 'Trap', 'AddressNumberAndStreet', 'AddressAccuracy',
                 'Date', 'Species', 'Station']
    train.drop(filter_out, axis=1, inplace=True) # these features are not used in prediction
    return train

# Note: This is a POST-DEADLINE improvement
# This procedure calculates the number of duplicated rows by ['Date', 'Species'], ['Date', 'Trap'], ['Date', 'Block'],
# ['Year', 'Species'], and etc. This improves LB score by about 0.017-0.018
def AddMoreFeatures(df, list_cols):
    indicator = np.not_equal(df.N_Dupl.values, np.ones(df.shape[0]))
    df['N'] = indicator
    grouped = df.groupby(by=list_cols, as_index=False)['N'].sum()
    grouped.columns = [list_cols[0], list_cols[1], 'N_Dupl_%s_%s' %(list_cols[0], list_cols[1])]
    df.drop('N', axis=1, inplace=True)
    result = df.merge(grouped, on=list_cols, how="left")
    return result
def shuffle(X, y, seed=1337):
    np.random.seed(seed)
    shuffle = np.arange(len(y))
    np.random.shuffle(shuffle)
    X = X.iloc[:,shuffle]
    y = y.iloc[:,shuffle]
    return X, y

########################
def train_nn_classification_model(
    learning_rate,
    steps,
    batch_size,
    hidden_units,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
  """
  In addition to training, this function also prints training progress information,
  a plot of the training and validation loss over time, as well as a confusion
  matrix.

  Args:
    learning_rate: An `int`, the learning rate to use.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    hidden_units: A `list` of int values, specifying the number of neurons in each layer.
    training_examples: A `DataFrame` containing the training features.
    training_targets: A `DataFrame` containing the training labels.
    validation_examples: A `DataFrame` containing the validation features.
    validation_targets: A `DataFrame` containing the validation labels.

  Returns:
    The trained `DNNClassifier` object.
  """

  periods = 10
  steps_per_period = steps / periods

  # Create input functions
  training_input_fn = learn_io.pandas_input_fn(
     x=training_examples, y=training_targets,
     num_epochs=None, batch_size=batch_size)

  predict_training_input_fn = learn_io.pandas_input_fn(
     x=training_examples, y=training_targets,
     num_epochs=1, shuffle=False)

  predict_validation_input_fn = learn_io.pandas_input_fn(
      x=validation_examples, y=validation_targets,
      num_epochs=1, shuffle=False)

  # Create a linear classifier object.
  feature_columns = set([tf.contrib.layers.real_valued_column(my_feature) for my_feature in training_examples])

  classifier = tf.contrib.learn.DNNClassifier(
      feature_columns=feature_columns,
      n_classes=2,
      hidden_units=hidden_units,
      optimizer=tf.train.AdagradOptimizer(learning_rate=learning_rate),
      gradient_clip_norm=5.0,
      config=tf.contrib.learn.RunConfig(keep_checkpoint_max=1)
  )

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print "Training model..."
  print "Error (on validation data):"
  training_errors = []
  validation_errors = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    classifier.fit(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # Take a break and compute predictions.
    training_predictions = np.array(list(classifier.predict_proba(input_fn=predict_training_input_fn)))
    validation_predictions = np.array(list(classifier.predict_proba(input_fn=predict_validation_input_fn)))
    #print training_predictions

    # Compute training and validation errors.
    #training_log_loss = metrics.log_loss(training_targets, training_predictions)
    #validation_log_loss = metrics.log_loss(validation_targets, validation_predictions)
    training_roc = metrics.roc_auc_score(training_targets, training_predictions[:,1])
    validation_roc = metrics.roc_auc_score(validation_targets, validation_predictions[:,1])
    # Occasionally print the current loss.
    print "  period %02d : %0.2f" % (period, validation_roc)
    # Add the loss metrics from this period to our list.
    training_errors.append(training_roc)
    validation_errors.append(validation_roc)
    '''
  print "Model training finished."
  # Remove event files to save disk space.
  #_ = map(os.remove, glob.glob(os.path.join(classifier.model_dir, 'events.out.tfevents*')))

  # Calculate final predictions (not probabilities, as above).
  #final_predictions = list(classifier.predict(validation_examples))
  #accuracy = metrics.accuracy_score(validation_targets, final_predictions)
  #print "Final accuracy (on validation data): %0.2f" % accuracy

  # Output a graph of loss metrics over periods.
  plt.ylabel("Errors")
  plt.xlabel("Periods")
  plt.title("Errors vs. Periods")
  plt.plot(training_errors, label="training")
  plt.plot(validation_errors, label="validation")
  plt.legend()
  plt.show()

  # Output a plot of the confusion matrix.
  cm = metrics.confusion_matrix(validation_targets, final_predictions)
  # Normalize the confusion matrix by row (i.e by the number of samples
  # in each class)
  cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
  ax = sns.heatmap(cm_normalized, cmap="bone_r")
  ax.set_aspect(1)
  plt.title("Confusion matrix")
  plt.ylabel("True label")
  plt.xlabel("Predicted label")
  plt.show()
  '''
  return classifier
###############
#######################################
Data_Version = 2 # 2    Choose the data version (1,2)
train = LoadTrain(Data_Version)  # load prepared data
train  = train.astype(float)
train = train.rename(columns={'CULEX ERRATICUS': 'ERRATICUS', 'CULEX PIPIENS': 'PIPIENS','CULEX PIPIENS/RESTUANS' : 'PIPIENSRESTUANS', 'CULEX RESTUANS': 'RESTUANS',  'CULEX SALINARIUS': 'SALINARIUS', 'CULEX TARSALIS':'TARSALIS',  'CULEX TERRITANS':'TERRITANS'})
#print train.describe()
#print train.count
#print train.head

print train.iloc[:,:10]
print train.iloc[:,10:20]
print train.iloc[:,20:30]
print train.iloc[:,31:40]
print train.iloc[:,41:60]
print train.iloc[:,60:80]
print train.iloc[:,80:96]
#train = train.reindex(np.random.permutation(train.index))
wnv=train['WnvPresent']#.values    # prediction target
#wnv=train.WnvPresent.values    # prediction target
#print wnv
train.drop(['NumMosquitos', 'WnvPresent'], axis=1, inplace=True)
#train, wnv = shuffle(train, wnv)
#print train.count
#train=train.iloc[:,:89]
#test = LoadTest(Data_Version)
#print test.describe()
#print test.count

nb_folds = 2
kfolds = KFold(len(wnv), nb_folds, shuffle=True, random_state=1337)
av_roc = 0.
f = 0

print kfolds


#    To train the DNN classifier with features and target data
#    :param features:
#    :param target:
#    :return: trained  classifier

for trainrows, validrows in kfolds:
    print('---'*20)
    print('Fold', f)
    print('---'*20)
    f += 1
    train_rows = train.iloc[trainrows,:]
    train_validation = train.iloc[validrows,:]
    wnv_train = wnv.iloc[trainrows]
    wnv_validation = wnv.iloc[validrows]

    classifier = train_nn_classification_model(
    learning_rate=0.05,
    steps=1000,
    batch_size=20,
    hidden_units=[10, 10],
    training_examples=train_rows,
    training_targets=wnv_train,
    validation_examples=train_validation,
    validation_targets=wnv_validation)

#################
print("Generating submission...")
classifier = train_nn_classification_model(
learning_rate=0.05,
steps=1000,
batch_size=20,
hidden_units=[10, 10],
training_examples=train,
training_targets=wnv,
validation_examples=train_validation,
validation_targets=wnv_validation)
trained_model=classifier;
###########
test = LoadTest(Data_Version)
#print test.describe()
ids=test['Id']
test.drop(['Id'], axis=1, inplace=True)
test  = test.astype(float)
test = test.rename(columns={'CULEX ERRATICUS': 'ERRATICUS', 'CULEX PIPIENS': 'PIPIENS','CULEX PIPIENS/RESTUANS' : 'PIPIENSRESTUANS', 'CULEX RESTUANS': 'RESTUANS',  'CULEX SALINARIUS': 'SALINARIUS', 'CULEX TARSALIS':'TARSALIS',  'CULEX TERRITANS':'TERRITANS'})
print test.count

print test.iloc[:,:10]
print test.iloc[:,10:20]
print test.iloc[:,20:30]
print test.iloc[:,31:40]
print test.iloc[:,41:60]
print test.iloc[:,60:80]
print test.iloc[:,80:95]

predict_test_input_fn = learn_io.pandas_input_fn(
     x=test, y=ids,
     num_epochs=1, shuffle=False)

preds = np.array(list(trained_model.predict_proba(input_fn=predict_test_input_fn)))
#preds = trained_model.predict_proba(test)
print preds
