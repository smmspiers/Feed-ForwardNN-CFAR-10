import numpy as np


def minmax_initialisation(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def get_pca_matrix(data, pov=0.9):
    # Compute covariance matrix
    covariance = np.cov(data, rowvar=False)
    # Calculate eigen values and vectors
    eigen_vals, eigen_vecs = np.linalg.eig(covariance)
    # Get proportion of variance (pov) for each component
    percens_of_var = eigen_vals / np.sum(eigen_vals)
    # Calculate cumulative percentages
    cum_percens_of_var = np.cumsum(percens_of_var)
    # Calculate num of principal components which give desired pov
    n_princ = np.count_nonzero(cum_percens_of_var < pov)
    # Sort eigenvalues according to sorted eigenvalues
    idx = np.argsort(eigen_vals)[::-1]
    eigen_vecs = eigen_vecs[:, idx]
    princ_eigh = eigen_vecs[:, :n_princ]
    return princ_eigh


def reduce_dimensions(data, pca_matrix):
    # subtract mean from all data points
    data -= np.mean(data)
    # project points onto PCA axes
    return np.dot(data, pca_matrix)
