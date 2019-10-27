import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_distances

# import data
ratings = pd.read_csv('rtg.csv')
# import kolom ke variabel
weight = ratings[['userId', 'movieId', 'S']]
# plot isi data ke matrix
llor = weight.pivot(index='userId', columns='movieId', values='S').fillna(0)
llor_matrix = csr_matrix(llor.values)
llor_array = llor.values
# menghitung user similarity
user_similarity = pairwise_distances(llor_array, metric='cosine')

# menghitung user 0
K = 60
pred_ratings = np.zeros(llor_matrix.shape)
for idx_user in range(pred_ratings.shape[0]):
    for idx_object in range(pred_ratings.shape[1]):
        print("Menghitung user ke -", idx_user, "dan objek ke - ", idx_object)
        # mengambil similarity untuk ke user index_user
        selected_similar = user_similarity[idx_user, :]
        # urutkan similarity dari terkecil hingga terbesar
        selected_similar, idx_sorted = np.sort(selected_similar), np.argsort(selected_similar)
        # hilangkan index pertama pada similarity yang telah diurutkan
        selected_similar, idx_sorted = np.delete(selected_similar, 0), np.delete(idx_sorted, 0)
        # ambil nilai sebanyak K untuk mendapatkan similarity paling terdekat
        selected_similar, idx_sorted = selected_similar[:K], idx_sorted[:K]
        # ambil rating dari setiap user yang terdekat
        rating_similar_user = llor_array[idx_sorted,:]
        average_similar_rating = rating_similar_user.mean(axis=1)
        # hitung nilai prediksi untuk user idx_user terhadap objek idx_object
        average_rating_user = llor_array[idx_user,:].mean()
        pred_ratings[idx_user, idx_object] = average_rating_user + (np.sum(selected_similar * (rating_similar_user[:,idx_object] - average_similar_rating)) / np.sum(np.abs(selected_similar)))