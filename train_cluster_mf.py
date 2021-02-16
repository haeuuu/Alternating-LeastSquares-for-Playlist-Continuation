# TODO : preprocessing 코드 넣기

import pandas as pd
import numpy as np
from itertools import chain
from scipy import sparse
from implicit.als import AlternatingLeastSquares as ALS
from Title_Based_Playlist_Generator import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans

def flatten_series(nested_series, index):
    repeated_index = np.repeat(index, nested_series.map(len))
    return pd.DataFrame(chain.from_iterable(nested_series), index=repeated_index)

class ClusterBasedRecommender:
    def __init__(self, train_path, test_path, song_meta_path):
        self.train = pd.read_json(train_path, typ='frame')
        self.test = pd.read_json(test_path, typ='frame')
        self.train_test = pd.concat([self.train, self.test])
        self.train_test.reset_index(inplace=True)
        self.song_meta = pd.read_pickle(song_meta_path, typ = 'frame') # song_meta에서 freq theshold 이상인 tag만 걸러 'new_tags'에 저장해야함.

    def build_vocab(self):
        tags = pd.unique(list(chain.from_iterable(self.song_meta.new_tags)))

        self.sid_to_title = dict(pd.Series(self.song_meta.song_name, index=self.song_meta.id))
        self.sid_to_cid = dict(zip(self.song_meta.id, self.song_meta.new_song_id))

        self.tag_to_tid = {tags[i]: self.n_clusters + i for i in range(len(tags))}
        self.tid_to_tag = {tid: tag for tag, tid in self.tag_to_tid.items()}

        self.ply_to_title = {self.train_test.id[i]: self.train_test.plylst_title[i] for i in range(len(self.train_test))}

        # cid_to_song(sorted by frequency)
        song_freq = pd.DataFrame(pd.Series(chain.from_iterable(self.train.songs)).value_counts(), columns=["freq"])
        song_freq['cid'] = [self.sid_to_cid[i] for i in song_freq.index.tolist()]
        self.song_freq_dict = dict(
            song_freq.sort_values(['cid', 'freq'], ascending=False).groupby('cid').apply(lambda x: x.index.tolist()))

        # cluster id 추가
        self.train_test['cid'] = self.train_test.songs.map(lambda x: [self.sid_to_cid[i] for i in x])
        self.train_test['tid'] = self.train_test.tags.map(lambda x: [self.tag_to_tid[i] for i in x if i in tags])
        self.train_test['clusters_tags'] = self.train_test.cid + self.train_test.tid

    def build_rating_matrix(self):
        cl_tg = flatten_series(self.train_test.clusters_tags, self.train_test.id).reset_index().rename(columns={0: 'iid'})
        cl_tg = cl_tg.groupby(cl_tg.columns.tolist()).size().reset_index().rename(columns={0: 'rating'})

        # csr matrix 만들기
        self.uid_to_index = {j: i for i, j in enumerate(self.train_test.id.values)}
        self.index_to_uid = {j: i for i, j in self.uid_to_index.items()}

        user_index = cl_tg.id.map(self.uid_to_index).values
        item_index = cl_tg.iid.values
        user_item_matrix = sparse.coo_matrix((cl_tg.rating, (user_index, item_index)))

        return user_item_matrix

    def ALS(self, user_item_matrix, factors=512, regularization=0.08, iterations=50):
        print('Train ALS ...')
        self.als = ALS(factors=factors, regularization=regularization, iterations=iterations)  # use_gpu=True(default), iter 15(default)
        self.als.fit(user_item_matrix.T * 10)

    def build_item2vec(self):
        item_embedding = []
        iids = []

        for sid, cid in self.sid_to_cid.items():
            iids.append(str(sid))
            item_embedding.append(self.als.item_factors[cid])

        for tid in self.tid_to_tag.keys():
            iids.append(tid)
            item_embedding.append(self.als.item_factors[tid])

        self.song2vec = WordEmbeddingsKeyedVectors(self.als.item_factors.shape[1])
        self.song2vec.add(iids, item_embedding)


if __name__ == '__main__':
    rec = ClusterBasedRecommender(train_path, test_path, song_meta_path)
    pca_results = rec.pca(indicator_matrix_path=indicator_matrix_path, n_components = 256)
    rec.kmeans(pca_results = pca_results, n_clusters = 1000)
    rec.build_vocab()
    user_item_matrix = rec.build_rating_matrix()
    rec.als(user_item_matrix = user_item_matrix, factors = 512)
    rec.build_item2vec()