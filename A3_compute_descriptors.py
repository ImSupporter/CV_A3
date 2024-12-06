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

# 2000 * 512(channel)
def max_pooling(cnn_list):
    result = []
    for cnn in cnn_list:
        row_max = np.max(cnn, axis=1)

        # normalization
        row_norms = np.linalg.norm(row_max)
        if row_norms == 0:
            row_norms = 1.0
        row_max = row_max/row_norms

        result.append(row_max)
    return np.array(result)

def mean_pooling(cnn_list):
    result = []
    for cnn in cnn_list:
        row_mean = np.mean(cnn, axis=1)

        # normalization
        row_norms = np.linalg.norm(row_mean)
        if row_norms == 0:
            row_norms = 1.0
        row_mean = row_mean/row_norms

        result.append(row_mean)
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

    # CNN Feature 불러오기
    cnn_list = []
    for path in cnn_files:
        with open(cnn_dir+path,'rb') as f:
            data = f.read()
            arr = np.frombuffer(data, dtype=np.float32)
            cnn = arr.reshape((-1,512)).T
            cnn_list.append(cnn)
    cnn_list = np.array(cnn_list)
    
    # 각 이미지에 대한 cnn(D=512)의 크기가 1.
    cnn_max = max_pooling(cnn_list)
    cnn_mean = mean_pooling(cnn_list)

    print('max pooling:', cnn_max.shape)
    print('mean pooling:', cnn_mean.shape)
    
    # kmeans centers 구하기(24개)
    # kmeans = KMeans(n_clusters=24, random_state=0).fit(vec_list)
    # print(kmeans.cluster_centers_.shape)
    # np.save('sift_centers', kmeans.cluster_centers_)

    # centers 불러오기
    centers = np.load('./sift_centers.npy')
    vlad = VLAD(vec_list,vec_nums, centers)
    cnn_max *= 8 # cnn_max pooling에 대한 가중치
    cnn_mean *= 1 # cnn_mean pooling에 대한 가중치
    des_mine = np.c_[vlad, cnn_max, cnn_mean]
    des_mine = des_mine.astype(np.float32)
    N, D = des_mine.shape
    print('N,D:',N,D)
    with open("A3_2018312081.des", 'wb') as f:
        f.write(np.array([N,D], dtype=np.int32).tobytes())
        f.write(des_mine.tobytes())




