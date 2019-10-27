import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_distances

# ~IMPORT DATA~
ratings = pd.read_csv('rtg.csv')
movies = pd.read_csv('movies.csv')

# ~READ DATA~
movie_data = movies[['movieId', 'title']]
movie_data.head()

llor_matrix = ratings[['userId', 'movieId', 'S']]
llor_matrix.head()

merge_book_ratings = pd.merge(llor_matrix, movie_data, on=['movieId'], how='inner')
merge_book_ratings

# ~OBSERVASI DATA~
# Menghitung banyaknya konten id yang paling sering di rating oleh pengguna
rating_count = pd.DataFrame(llor_matrix.groupby('movieId')['S'].count())
rating_count.sort_values('S', ascending=False).head()

# Menghitung rata2 nilai S dari data
average_rating = pd.DataFrame(llor_matrix.groupby('movieId')['S'].mean())
average_rating['ratingCount'] = pd.DataFrame(llor_matrix.groupby('movieId')['S'].count())
average_rating.sort_values('ratingCount', ascending=False).head()

# Ubah data menjadi matrix 2D untuk diproses menggunakan kNN
ratings_pivot = llor_matrix.pivot(index='movieId', columns='userId', values='S').fillna(0)
ratings_pivot_matrix = csr_matrix(ratings_pivot.values)
print(ratings_pivot.head())

# ~MENGHITUNG SIMILARITY USER-USER~
# Konversi data menjadi numpy array
ratings_data = ratings_pivot.T
ratings_data_array = ratings_data.values

# Mendapatkan similarity dari tiap user-user
user_similarity = pairwise_distances(ratings_data_array, metric='cosine')
print(pd.DataFrame(user_similarity).head())

# menghitung user 0
K = 2
pred_ratings = np.zeros(K)
# mengambil similarity paling kecil dari user 0
selected_similar = user_similarity[0][:]
sorted_similar, index_sorted = np.sort(selected_similar), np.argsort(selected_similar)
most_similar, index_similar = sorted_similar[:K], index_sorted[:K]

# mengambil rating user
ratings_similar_user = ratings_data_array[index_similar,:]
# hitung rata-rata dari tiap user
mean_user_rating = ratings_similar_user.mean(axis=1)
diff_ratings = ratings_similar_user - mean_user_rating[:, np.newaxis]

# menghitung prediksi untuk user 0 terhadap object 1
pred = mean_user_rating[0] + np.dot(most_similar * diff_ratings[:,2]) / np.dot(most_similar)
print(pred)
pd.DataFrame(diff_ratings)

