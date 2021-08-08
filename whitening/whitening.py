import numpy as np


def compute_kernel_bias(vecs):
    """计算kernel和bias
    最后的变换：y = (x + bias).dot(kernel)
    """
    vecs = np.concatenate(vecs, axis=0)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    # return None, None
    # return W, -mu
    return W[:, :256], -mu


embeddings = np.load('whitening/embedding_avg.npy')
kernel, bias = compute_kernel_bias(embeddings)