import numpy as np
import pandas as pd 
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_distances

# import data
ratings = pd.read_csv('rtg.csv')

# import kolom ke variabel
weight = ratings[['userId', 'movieId', 'S']]

# plot isi data ke matrix
llor = weight.pivot(index='userId', columns='movieId', values='S').fillna(0)
llor_matrix = csr_matrix(llor.values)

print(llor)
