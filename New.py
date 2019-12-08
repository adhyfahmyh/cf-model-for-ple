import operator
import numpy as np
import pandas as pd 
from numpy import nan
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_distances
from correlation_pearson.code import CorrelationPearson
pearson = CorrelationPearson()

# import data
raw = pd.read_csv('WLO_raw(1).csv', delimiter=';')

# preprocessing
raw = raw[pd.notnull(raw['rating'])]
raw = raw[pd.notnull(raw['timespent'])]
where_are_NaNs = np.isnan(raw)
raw[where_are_NaNs] = 0

# variables
E = raw['rating']
a = raw['bookmarked']
t = raw['timespent']
c = raw['total_selection']
b = np.exp(-t)

# normalized
normalize = lambda x : ((x - np.min(x))/(np.max(x)-np.min(x)))
A = normalize(a)
B = normalize(b)
C = normalize(c)

# weight learning object
e = E
i = A + ((2)*(B)) + ((2)*(C)*(E))
S = (1/2)*(e+i)

user_id = raw[['user_id']]
content_id = raw[['content_id']]
d = {
    'A' : A, 'B' : B, 'C' : C,
    'implicit' : i, 'explicit' : e, 'S' : S
}
data = pd.DataFrame(data = d)

joinraw = raw.join(data)
weight = joinraw[['user_id', 'content_id', 'S']]

# llor matrix
llor = weight.pivot_table(index='user_id', columns='content_id', values='S').fillna(0)
llor_matrix = csr_matrix(llor.values)
llor_array = llor.values

#similarities
dict_x={}
for i in range(len(llor_array)):
    dict_x[i]={}
    for j in range(len(llor_array)):
        if i==j:
            continue
        else:
            dict_x[i][j]= pearson.result(llor_array[i],llor_array[j])

# predict value
dict_x={}
k=10
for i in range(len(llor_array)):
    print("=========INI USER KE- ",i,"=================")
    dict_x[i]={}
    temp={}
    for j in range(len(llor_array)):
        if i==j:
            continue
        else:
            temp[j]= pearson.result(llor_array[i],llor_array[j])
    tmp = {k: temp[k] if not np.isnan(temp[k]) else 0 for k in temp}
#     dict_x[i] = dict(sorted(tmp.items(), key=operator.itemgetter(1),reverse=True)[:10])
    tmp = dict(sorted(tmp.items(), key=operator.itemgetter(1),reverse=True)[:k])
    pearsonDF = pd.DataFrame.from_dict(tmp, orient='index')
    pearsonDF.columns = ['similarityIndex']
    pearsonDF['user_id'] = pearsonDF.index
    pearsonDF.index = range(len(pearsonDF))
    mean_rating = [llor_array[y].mean() for y in list(pearsonDF['user_id'])]
    pearsonDF['ave_rating'] = mean_rating
#     print(pearsonDF)
    topUsersRating=pearsonDF.merge(weight, left_on='user_id', right_on='user_id', how='inner')
    topUsersRating['weight'] = topUsersRating['S'] - topUsersRating['ave_rating']
    topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['weight']
#     print(topUsersRating)
    tempTopUsersRating = topUsersRating.groupby('content_id').sum()[['similarityIndex','weightedRating']]
    tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']
#     print(tempTopUsersRating)
    recommendation_df = pd.DataFrame()
    recommendation_df['weighted average recommendation score'] = llor_array[i].mean()+(tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex'])
    recommendation_df['content_id'] = tempTopUsersRating.index
#     print(recommendation_df)
    recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
    print(recommendation_df)
