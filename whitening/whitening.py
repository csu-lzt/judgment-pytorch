'''
第二步，将第一步得到的句子向量做一个白化操作
修改embeddings和save_path
'''
import numpy as np
import time


def compute_kernel_bias(vecs):
    """计算kernel和bias
    vecs.shape = [num_samples, embedding_size]，
    最后的变换：y = (x + bias).dot(kernel)
    """
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W, -mu


def transform_and_normalize(vecs, kernel, bias):
    """应用变换，然后标准化
    """
    vecs = (vecs + bias).dot(kernel)
    # return vecs / (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5
    return vecs.astype(np.float32)


start = time.time()
embeddings = np.load('whitening/embedding_cls.npy')
print('加载完成=====', embeddings.shape)
kernel, bias = compute_kernel_bias(embeddings)
end1 = time.time()
print('耗时1=====', end1 - start)
embeddings_whiten = transform_and_normalize(embeddings, kernel, bias)
end2 = time.time()
print('耗时2=====', end2 - end1)
print(embeddings_whiten.shape)
save_path = 'whitening/embedding_cls_whiten.npy'
np.save(save_path, embeddings_whiten)
