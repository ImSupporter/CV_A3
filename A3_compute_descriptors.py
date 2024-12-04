# 2000개의 사진 및 description
# SIFT - 2000 * 128 (uint8)
# CNN - 2000 * 14 * 14 * 512 (float32)
# -> 2000 * D (단 D <= 4096)

# 목표 3.7?
import numpy as np
import os

from setuptools.namespaces import flatten
from sklearn.cluster import KMeans



def rank_cluster_centers(vec, centers):
    distances = np.linalg.norm(centers - vec, axis=1)
    return np.argsort(distances)

def VLAD(des_list, centers):
    result = []
    print(des_list.shape)
    for des in des_list:
        agg_vecs = np.zeros_like(centers)
        for vec in des:
            predicted_idx = rank_cluster_centers(vec, centers)[0]
            agg_vecs[predicted_idx] = agg_vecs[predicted_idx] + (vec - centers[predicted_idx])
        row_norms = np.linalg.norm(agg_vecs, axis=1)
        row_norms[row_norms == 0] = 1.0
        agg_vecs = agg_vecs/row_norms.reshape((-1,1))
        result.append(agg_vecs.flatten())
    return np.array(result)


if __name__ == '__main__':
    # feature(CNN) 불러오기
    cnn_dir = './features/cnn/'
    cnn_files = os.listdir(cnn_dir)
    cnn_files = sorted(cnn_files,key=lambda x:int(x[:4]))
    des_list=[]
    for path in cnn_files:
        with open(cnn_dir+path, 'rb') as f:
            data = f.read()
            arr = np.frombuffer(data, dtype=np.float32)
            des = arr.reshape((196, 512)).T
            des_list.append(des)
    des_list = np.array(des_list)

    # # vector 추출,이미지 당 h*w(196) 크기의 벡터 512개 => 1024000
    # vectors = []
    # for des in des_list:
    #     for v in des:
    #         vectors.append(v)
    # print(vectors[0].shape)
    #
    # # kmeans centers 구하기(20개)
    # kmeans = KMeans(n_clusters=20, random_state=0).fit(vectors)
    # print(kmeans.cluster_centers_)
    # np.save('centers', kmeans.cluster_centers_)

    # centers 불러오기
    centers = np.load('./centers.npy')

    des_mine = VLAD(des_list, centers)
    des_mine = des_mine.astype(np.float32)
    N, D = des_mine.shape
    print(N,D)
    with open("A3_2018312081.des", 'wb') as f:
        f.write(np.array([N,D], dtype=np.int32).tobytes())
        f.write(des_mine.tobytes())




