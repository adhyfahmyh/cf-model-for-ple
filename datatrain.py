import numpy as np
import os
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

read_movies = pd.read_csv('ml-latest-small/movies.csv')
read_ratings = pd.read_csv('ml-latest-small/ratings.csv')
E = read_ratings['rating']

a = np.random.randint(0, 2, size = len(read_ratings))
t = np.random.randint(0,3601, size = len(read_ratings))
c = np.random.randint(0, 11, size = len(read_ratings))
b = np.exp(-t)

normalize = lambda x : ((x - np.min(x))/(np.max(x)-np.min(x)))
normC = normalize(c)

implicit = a + (2*b) + (2*normC*E)
S = 0.5*(E+implicit)

d = {
    'bookmark (A)' : a, 'time' : t, '(B)' : b, 'norm (C)' : normC,
    'implicit' : implicit, 'S' : S
}
df = pd.DataFrame(data = d)

bmax = np.max(c)

joindf = read_ratings.join(df)
joindf.to_csv('rtg.csv', sep=',')

check = pd.read_csv('rtg.csv',
    usecols=['userId', 'movieId', 'S'],
    dtype = {'userId' : 'int32', 'movieId' : 'int32', 'S' : 'float'}
)

features = check.pivot(
    index = 'userId',
    columns = 'movieId',
    values = 'S'
).fillna(0)

mat_features = csr_matrix(features.values)

model = NearestNeighbors(metric='cosine', algorithm='brute',
n_neighbors=20, n_jobs=-1)

model.fit(mat_features)
