import numpy as np
import pandas as pd 
from numpy import nan
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_absolute_error
from correlation_pearson.code import CorrelationPearson
pearson = CorrelationPearson()
import operator

# Import data
raw = pd.read_csv('demo_sidang.csv', delimiter=';')

# Cleaning/preprocessing
# raw = raw[pd.notnull(raw['rating'])]
# raw = raw[pd.notnull(raw['timespent'])]
where_are_NaNs = np.isnan(raw)
raw[where_are_NaNs] = 0
# print(raw)
pivot_raw = raw.pivot_table(index='user_id', columns='content_id', values='rating').fillna(0)

# raw.to_csv('raw.csv', sep=',')


# Variables
E = raw['rating']
a = raw['bookmarked']
t = raw['timespent']
c = raw['total_selection']
b = np.exp(-t)

# Normalize
normalize = lambda x : ((x - np.min(x))/(np.max(x)-np.min(x)))
A = normalize(a)
B = normalize(b)
C = normalize(c)
# print ((A.max()-A.min()))
# print ((B.max()-B.min()))
# print ((B.max()-B.min()))

# Weight Learning Object
e = E
i = A + ((2)*(B)) + ((2)*(C)*(E))
# print(i)
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

# LLOR matrix
llor = weight.pivot_table(index='user_id', columns='content_id', values='S').fillna(0)
llor_matrix = csr_matrix(llor.values)
llor_array = llor.values
# print(llor)

# Similarities
dict_x={}
for i in range(len(llor_array)):
    dict_x[i]={}
    for j in range(len(llor_array)):
        if i==j:
            continue
        else:
            dict_x[i][j]= pearson.result(llor_array[i],llor_array[j])

# Prediction
dict_x={}
final_score=[]
final_seq=[]
k=10
for i,value_i in enumerate(list(llor.index)):
    # print("=========INI USER ID: ",value_i,"=================")
    dict_x[i]={}
    temp={}
    for j,value_j in enumerate(list(llor.index)):
        if i==j:
            continue
        else:
            temp[j]= pearson.result(llor_array[i],llor_array[j])
    tmp = {key: temp[key] if not np.isnan(temp[key]) else 0 for key in temp}
    tmp = dict(sorted(tmp.items(), key=operator.itemgetter(1),reverse=True)[:k])
    pearsonDF = pd.DataFrame.from_dict(tmp, orient='index')
    pearsonDF.columns = ['similarityIndex']
    pearsonDF['user_id'] = pearsonDF.index
    pearsonDF.index = range(len(pearsonDF))
    mean_rating = [llor_array[y].mean() for y in list(pearsonDF['user_id'])]
    pearsonDF['ave_rating'] = mean_rating
    topUsersRating=pearsonDF.merge(weight, left_on='user_id', right_on='user_id', how='inner')
    topUsersRating['weight'] = topUsersRating['S'] - topUsersRating['ave_rating']
    topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['weight']
    tempTopUsersRating = topUsersRating.groupby('content_id').sum()[['similarityIndex','weightedRating']]
    tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']
    recommendation_df = pd.DataFrame()
    recommendation_df['recommendation score'] = llor_array[i].mean()+(tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex'])
    recommendation_df['content_id'] = tempTopUsersRating.index
    recommendation_df = recommendation_df.sort_values(by='recommendation score', ascending=False)
    for index, row in recommendation_df.iterrows():
        final_score.append([value_i,row['content_id'],row['recommendation score']])
    final_seq.append([value_i,list(recommendation_df["content_id"])])
    # print(recommendation_df)

# Recommendation result table
final_score_df = pd.DataFrame(final_score,columns=["user_id","content_id","Recommendation Score"])
# print(final_score_df)

# LLOP matrix
llop = final_score_df.pivot_table(index='user_id', columns='content_id', values='Recommendation Score').fillna(0)
# print(llop)

# Data Final
final_seq_df = pd.DataFrame(final_seq,columns=["user_id","Recommendation Sequence"])
# print(final_seq_df)
