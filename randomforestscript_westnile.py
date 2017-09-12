'''
    This code gets a ROC AUC score (local 4-fold validation)
    in the Kaggle Nile virus prediction challenge.
    Classifier used is Random Forest classifier
'''
import numpy as np
import csv

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn import metrics
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler



# let's define some utility functions
# parse weather data
def get_weather_data():
    weather_dic = {}
    fi = csv.reader(open("weather.csv"))
    weather_head = fi.next()
    for line in fi:
        if line[0] == '1':
            continue
        weather_dic[line[1]] = line
    weather_indexes = dict([(weather_head[i], i) for i in range(len(weather_head))])
    return weather_dic, weather_indexes

# process each line in other data
def process_line(line, indexes, weather_dic, weather_indexes):
    def get(name):
        return line[indexes[name]]

    date = get("Date")
    month = float(date.split('-')[1])
    week = int(date.split('-')[1]) * 4 + int(date.split('-')[2]) / 7
    latitude = float(get("Latitude"))
    longitude = float(get("Longitude"))
    tmax = float(weather_dic[date][weather_indexes["Tmax"]])
    tmin = float(weather_dic[date][weather_indexes["Tmin"]])
    tavg = float(weather_dic[date][weather_indexes["Tavg"]])
    dewpoint = float(weather_dic[date][weather_indexes["DewPoint"]])
    wetbulb = float(weather_dic[date][weather_indexes["WetBulb"]])
    pressure = float(weather_dic[date][weather_indexes["StnPressure"]])

    return [month, week, latitude, longitude, tmax, tmin, tavg, dewpoint, wetbulb, pressure]

# preprocess data - normalize
def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

# shuffle the rows
def shuffle(X, y, seed=1337):
    np.random.seed(seed)
    shuffle = np.arange(len(y))
    np.random.shuffle(shuffle)
    X = X[shuffle]
    y = y[shuffle]
    return X, y




# now the data processing script

print("Processing training data...")

rows = []
labels = []
fi = csv.reader(open("train.csv"))
head = fi.next()
indexes = dict([(head[i], i) for i in range(len(head))])
weather_dic, weather_indexes = get_weather_data()
for line in fi:
    rows.append(process_line(line, indexes, weather_dic, weather_indexes))
    labels.append(float(line[indexes["WnvPresent"]]))

X = np.array(rows)
y = np.array(labels)

X, y = shuffle(X, y)
X, scaler = preprocess_data(X)
#Y = np_utils.to_categorical(y)



print X
print y
print np.count_nonzero(y)*1.0/len(y)
input_dim = X.shape[1]
output_dim = 2

print("Validation...")

nb_folds = 4
kfolds = KFold(len(y), nb_folds)
av_roc = 0.
f = 0

"""
    To train the random forest classifier with features and target data
    :param features:
    :param target:
    :return: trained random forest classifier
    """
for train, valid in kfolds:
    print('---'*20)
    print('Fold', f)
    print('---'*20)
    f += 1
    X_train = X[train]
    X_valid = X[valid]
    y_train = y[train]
    y_valid = y[valid]
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    trained_model=clf;
    print "Trained model :: ", trained_model
    valid_preds = trained_model.predict_proba(X_valid)
    print valid_preds

    valid_preds = valid_preds[:, 1]
    roc = metrics.roc_auc_score(y_valid, valid_preds)
    print("ROC:", roc)
    av_roc += roc

print('Average ROC:', av_roc/nb_folds)

print("Generating submission...")
clf = RandomForestClassifier()
clf.fit(X, y)
trained_model=clf;
fi = csv.reader(open("test.csv"))
head = fi.next()
indexes = dict([(head[i], i) for i in range(len(head))])
rows = []
ids = []
for line in fi:
    rows.append(process_line(line, indexes, weather_dic, weather_indexes))
    ids.append(line[0])
X_test = np.array(rows)
X_test, _ = preprocess_data(X_test, scaler)

preds = trained_model.predict_proba(X_test)

fo = csv.writer(open("randomforestoutput.csv", "w"), lineterminator="\n")
fo.writerow(["Id","WnvPresent"])
for i, item in enumerate(ids):
    fo.writerow([ids[i], preds[i][1]])
