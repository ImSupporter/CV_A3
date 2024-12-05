# 2000개의 사진 및 description
# SIFT - 2000 * 128 (uint8)
# CNN - 2000 * 14 * 14 * 512 (float32)
# -> 2000 * D (단 D <= 4096)

# 목표 3.7?
import numpy as np
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt



def rank_cluster_centers(vec, centers):
    distances = np.linalg.norm(centers - vec, axis=1)
    return np.argsort(distances)

def VLAD(vec_list, vec_nums, centers):
    vec_idx = 0
    result = []
    for vec_num in vec_nums:
        agg_vecs = np.zeros_like(centers)
        vec_cnt = 0
        while vec_cnt < vec_num:
            predicted_idx = rank_cluster_centers(vec_list[vec_idx], centers)[0]
            agg_vecs[predicted_idx] = agg_vecs[predicted_idx] + (vec_list[vec_idx] - centers[predicted_idx])
            vec_cnt = vec_cnt + 1
            vec_idx = vec_idx + 1

        # normalization
        row_norms = np.linalg.norm(agg_vecs, axis=1)
        row_norms[row_norms == 0] = 1.0
        agg_vecs = agg_vecs/row_norms.reshape((-1,1))

        result.append(agg_vecs.flatten())
    return np.array(result)


if __name__ == '__main__':
    # feature(CNN) 불러오기
    cnn_dir = './features/cnn/'
    sift_dir = './features/sift/'

    cnn_files = os.listdir(cnn_dir)
    cnn_files = sorted(cnn_files,key=lambda x:int(x[:4]))
    sift_files = os.listdir(sift_dir)
    sift_files = sorted(sift_files, key= lambda x: int(x[:4]))

    vec_list=[]
    vec_nums = []
    for path in sift_files:
        with open(sift_dir+path, 'rb') as f:
            data = f.read()
            arr = np.frombuffer(data, dtype=np.uint8)
            des = arr.reshape((-1, 128))
            vec_nums.append(des.shape[0])
            for vec in des:
                vec_list.append(vec)
    des_list = np.array(vec_list)
    print(len(vec_list[0]))

    # # inertia 구하기(K=4로 결정)
    # ks = range(1,10)
    # inertias = []

    # for k in ks:
    #     model = KMeans(n_clusters=k)
    #     model.fit(des_list)
    #     inertias.append(model.inertia_)
    #     print(k, inertias[-1])

    # # Plot ks vs inertias
    # plt.figure(figsize=(4, 4))

    # plt.plot(ks, inertias, '-o')
    # plt.xlabel('number of clusters, k')
    # plt.ylabel('inertia')
    # plt.xticks(ks)
    # plt.show()


    # # vector 추출,이미지 당 h*w(196) 크기의 벡터 512개 => 1024000
    # vectors = []
    # for des in des_list:
    #     for v in des:
    #         vectors.append(v)
    # print(vectors[0].shape)
    
    # kmeans centers 구하기(20개)
    # kmeans = KMeans(n_clusters=8, random_state=0).fit(vec_list)
    # print(kmeans.cluster_centers_.shape)
    # np.save('sift_centers', kmeans.cluster_centers_)

    # centers 불러오기
    centers = np.load('./sift_centers.npy')

    des_mine = VLAD(vec_list,vec_nums, centers)
    des_mine = des_mine.astype(np.float32)
    N, D = des_mine.shape
    print('N,D:',N,D)
    with open("A3_2018312081.des", 'wb') as f:
        f.write(np.array([N,D], dtype=np.int32).tobytes())
        f.write(des_mine.tobytes())




