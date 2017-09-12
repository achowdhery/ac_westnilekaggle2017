import pandas as pd
import numpy as np
import csv

# --- Read Data --- #
#header=['viewer_id','media_id','publisher_category','watched_pct']
header=['viewer_id', 'media_id']
df = pd.read_csv('jw_player_play_data.csv', low_memory=False)

n_users=df['viewer_id'].unique().shape[0]
n_items=df['media_id'].unique().shape[0]
print 'Number of viewers = '+str(n_users)+' |Number of media items = '+str(n_items);

media_items=df['media_id'].unique()
print len(media_items)
print media_items

viewers_list=df['viewer_id'].unique()
print len(viewers_list)
print viewers_list
#print df.head()

publisher_list=df['publisher_category'].unique()
print len(publisher_list)
print publisher_list

print df.tail()

for publisher in publisher_list[9:10]:
    viewer_subset=df.loc[df['publisher_category'] == publisher]
    #viewer_subset=viewer_subset[:100000]
    print publisher
    print 'Number of viewers: '+ str(len(viewer_subset['viewer_id'].unique()))
    print 'Number of media: '+ str(len(viewer_subset['media_id'].unique()))

    zero_data = np.zeros(shape=(len(viewer_subset['viewer_id'].unique()),len(viewer_subset['media_id'].unique())));
    df = pd.DataFrame(zero_data, columns=(viewer_subset['media_id'].unique()))
    df=df.set_index(viewer_subset['viewer_id'].unique())

    userItemMatrix={};
    for [iv,viewer] in enumerate(viewer_subset['viewer_id'].unique()):
        v_current=viewer_subset.loc[viewer_subset['viewer_id'] == viewer]


        if (iv%1000==0):
            print iv
        seen_set = set(); duplicate_set=set();
        viewengagement={};
        for i,x in enumerate(v_current['media_id']):
            viewengagement[x]=0;
            if x in seen_set or seen_set.add(x):
                duplicate_set.add(x);
                if v_current.iloc[i,9]>viewengagement[x]:
                    viewengagement[x]= v_current.iloc[i,9];
            else:
                viewengagement[x]=v_current.iloc[i,9];

        unique_set = seen_set - duplicate_set;
        for k in viewengagement.keys():
            userItemMatrix[viewer,k]=viewengagement[k]/100.0;
            df.loc[viewer,k]=viewengagement[k]/100.0;

    data_sims_temp=df.ix[:,:]

    from sklearn import cross_validation as cv
    train_data, test_data = cv.train_test_split(data_sims_temp, test_size=0.25)

    train_data_matrix = train_data.as_matrix()
    test_data_matrix = test_data.as_matrix()


    from sklearn.metrics.pairwise import pairwise_distances
    user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
        #print user_similarity

    #print user_similarity[:4, :4]
    def predict(ratings, similarity, type='user'):
        if type == 'user':
            mean_user_rating = ratings.mean(axis=1)
            #You use np.newaxis so that mean_user_rating has same format as ratings
            ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
            pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
            return pred


    user_prediction = predict(train_data_matrix, user_similarity, type='user')

    print user_prediction

    from sklearn.metrics import mean_squared_error
    from math import sqrt
    def rmse(prediction, ground_truth):
        prediction = prediction[ground_truth.nonzero()].flatten()
        ground_truth = ground_truth[ground_truth.nonzero()].flatten()
        return sqrt(mean_squared_error(prediction, ground_truth))

    print 'User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix))
