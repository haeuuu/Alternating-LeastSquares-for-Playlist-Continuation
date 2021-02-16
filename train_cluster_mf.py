import pandas as pd
import numpy as np
from itertools import chain
from scipy import sparse
from implicit.als import AlternatingLeastSquares as ALS
from Title_Based_Playlist_Generator import *

# train,val datag
tr = pd.read_json(train_path, typ = 'frame')
val = pd.read_json(val_que_path, typ = 'frame')
tr_val = pd.concat([tr,val])
tr_val.reset_index(inplace = True)

# train playlist를 이용해서 곡 별 태그(new_tags)와 cluster id( PCA + K-means 결과)를 추가한 song_meta
song_meta = pd.read_pickle(song_meta_cluster_path)
pca = pd.read_pickle(pca_path)
song_meta['new_song_id'] = pca['MiniBatchKmeans']

# dictionary 만들기
tags = pd.unique(list(chain.from_iterable(song_meta.new_tags)))

sid_to_title = dict(pd.Series(song_meta.song_name, index = song_meta.id))
sid_to_cid = dict(zip(song_meta.id, song_meta.new_song_id))
n_clusters = 1000 # k-means cluster 수

tag_to_tid = {tags[i]:n_clusters + i for i in range(len(tags))}
tid_to_tag = {tid:tag for tag,tid in tag_to_tid.items()}

ply_to_title = {tr_val.id[i]:tr_val.plylst_title[i] for i in range(len(tr_val))}

# cid_to_song(sorted by frequency)
song_freq = pd.DataFrame(pd.Series(chain.from_iterable(tr.songs)).value_counts(), columns = ["freq"])
song_freq['cid'] = [sid_to_cid[i] for i in song_freq.index.tolist()]
song_freq_dict = dict(song_freq.sort_values(['cid','freq'], ascending=False).groupby('cid').apply(lambda x:x.index.tolist()))

# cluster id 추가
tr_val['cid'] = tr_val.songs.map(lambda x:[sid_to_cid[i] for i in x])
tr_val['tid'] = tr_val.tags.map(lambda x:[tag_to_tid[i] for i in x if i in tags])
tr_val['clusters_tags'] = tr_val.cid + tr_val.tid

def flatten_series(nested_series, index):
    repeated_index = np.repeat(index, nested_series.map(len))
    return pd.DataFrame(chain.from_iterable(nested_series), index = repeated_index)

cl_tg = flatten_series(tr_val.clusters_tags, tr_val.id).reset_index().rename(columns = {0:'iid'})
cl_tg = cl_tg.groupby(cl_tg.columns.tolist()).size().reset_index().rename(columns = {0:'rating'})

# csr matrix 만들기
uid_to_index = {j:i for i,j in enumerate(tr_val.id.values)}
index_to_uid = {j:i for i,j in uid_to_index.items()}

user_index = cl_tg.id.map(uid_to_index).values
item_index = cl_tg.iid.values
user_item = sparse.coo_matrix((cl_tg.rating,(user_index, item_index)))

# train
als_model = ALS(factors = 512, regularization = 0.08, iterations = 50) # use_gpu=True(default), iter 15(default)
als_model.fit(user_item.T*10)

#
song_map = []
labels = []

for sid, cid in sid_to_cid.items():
    labels.append(str(sid))
    song_map.append(cluster_map[cid])
print(len(song_map))

for index, item in iid_to_items.items():
    if type(item) == str:
        labels.append(item)
        song_map.append(cluster_map[index])

print(len(song_map))

# Word2Vec-for-Recommendation > class Playlist2Vec를 활용하자.
song2vec = WordEmbeddingsKeyedVectors(cluster_map.shape[1])
song2vec.add(labels, song_map)

model = Playlist2Vec(train, val_que)
model.register_w2v(song2vec)
model.build_p2v()