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

from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt

'''
    Kaggle West Nile virus prediction challenge.

'''

# process weather data
def get_weather_data():
    weather_dic = {}
    fi = csv.reader(open("weather.csv"))
    weather_head = fi.next()
    for line in fi:
        if line[0] == '1': # keep 1 station
            continue
        weather_dic[line[1]] = line;
    weather_indexes = dict([(weather_head[i], i) for i in range(len(weather_head))])

    return weather_dic, weather_indexes

species_map = {'CULEX RESTUANS' : "100000",
              'CULEX TERRITANS' : "010000",
              'CULEX PIPIENS'   : "001000",
              'CULEX PIPIENS/RESTUANS' : "101000",
              'CULEX ERRATICUS' : "000100",
              'CULEX SALINARIUS': "000010",
              'CULEX TARSALIS' :  "000001",
              'UNSPECIFIED CULEX': "001000"}

def process_line(line, indexes, weather_dic, weather_indexes):
    def get(name):
        return line[indexes[name]]

    date = get("Date")
    month = float(date.split('-')[1])
    week = int(date.split('-')[1]) * 4 + int(date.split('-')[2]) / 7
    latitude = float(get("Latitude"))
    longitude = float(get("Longitude"))
    species=species_map[get("Species")];
    #traps=get("Trap");
    #print species, traps

    # weather features
    tmax = float(weather_dic[date][weather_indexes["Tmax"]])
    tmin = float(weather_dic[date][weather_indexes["Tmin"]])
    tavg = float(weather_dic[date][weather_indexes["Tavg"]])
    dewpoint = float(weather_dic[date][weather_indexes["DewPoint"]])
    wetbulb = float(weather_dic[date][weather_indexes["WetBulb"]])
    pressure = float(weather_dic[date][weather_indexes["StnPressure"]])
    temp=(weather_dic[date][weather_indexes["PrecipTotal"]])
    if (temp.isdigit()):
        if float(temp)>0:
            precipitation = 1.0;
        else:
            precipitation = 0.0;
    else:
        precipitation = 0.0;
    windspeed = float(weather_dic[date][weather_indexes["ResultSpeed"]])
    #return [month, week, latitude, longitude, tmax, tmin, tavg, dewpoint, wetbulb, pressure, precipitation, windspeed, species]

    return [month, week, latitude, longitude, tmax, tmin, tavg, dewpoint, wetbulb, pressure, windspeed, species]

def preprocess_data(X, scaler=None):
    #replace missing values
    df = pd.DataFrame(X);
    df = df.replace('T', -1)
    df = df.replace('M', -1)
    df = df.replace('-', -1)
    X = df.as_matrix()
    #print X[:,10]
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

def shuffle(X, y, seed=1337):
    np.random.seed(seed)
    shuffle = np.arange(len(y))
    np.random.shuffle(shuffle)
    X = X[shuffle]
    y = y[shuffle]
    return X, y




# now the actual script

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




#Sanity check the data - visualize, find correlations etc
#df = pd.DataFrame(X[:,0:9])
#pd.plotting.scatter_matrix(df);
#plt.show()
#print df.describe()

print "sample data: [month, week, latitude, longitude, tmax, tmin, tavg, dewpoint, wetbulb, pressure,  windspeed, species]"
print rows[:10]
print labels[:10]
featurelabels = {0 : "month",
           1 : "week",
           2 : "latitude",
           3 : "longitude",
           4 : "tmax",
           5 : "tmin",
           6 : "tavg",
           7 : "dewpoint",
           8 : "wetbulb",
           9 : "pressure",
           #10: "precipitation",
           10: "windspeed",
           11: "species",
}
print "Summary statistics for features"
def sanitycheckData(X, idx):
    print X[:,idx]
    df=pd.DataFrame(X[:,idx]);
    #print [x for x in X[:,0] if x<0]
    #df.hist()
    #plt.show()
    print df.describe()

for idx in xrange(len(featurelabels)):
    print idx, featurelabels[idx];
    sanitycheckData(X,idx)


# shuffle and preprocess_data
X, y = shuffle(X, y)
X, scaler = preprocess_data(X)
#Y = np_utils.to_categorical(y)


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
    clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1);
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
clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1);
clf.fit(X, y)
trained_model=clf;
##################
#Feature importance
importances = trained_model.feature_importances_
std = np.std([tree.feature_importances_ for tree in trained_model.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d %s (%f)" % (f + 1, indices[f], featurelabels[indices[f]],importances[indices[f]]))
print range(X.shape[1]), indices
# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="b", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), [featurelabels[k] for k in indices])
plt.xlim([-1, X.shape[1]])
plt.show()
###########################################
# Generate predictions
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
