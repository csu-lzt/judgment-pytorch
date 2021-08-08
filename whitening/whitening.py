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


start = time.time()
embeddings = np.load('whitening/embedding_avg.npy')
print('加载完成=====', embeddings.shape)
kernel, bias = compute_kernel_bias(embeddings)
end = time.time()
print('耗时=====', end - start)
