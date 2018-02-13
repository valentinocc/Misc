import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from time import time
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist

def MDS(dataset, reduced_dimension):
    
    distances = squareform(pdist(dataset, metric='euclidean'))
    N = distances.shape[1]
    """print(distances)
    for i in np.arange(N):
        print(np.mean(distances[:,i]))
        distances[:,i] = distances[:,i] - np.mean(distances[:,i])
    
    print(distances)"""
    
    J = np.identity(N) - 1/N * (np.ones((N,1)) @ np.ones((1,N)))
    
    B = -1/2 * J @ np.square(distances) @ J
    
    eigvals, eigvecs = np.linalg.eig(B)
    sorted_index = eigvals.argsort()[::-1]
    eigvals = ((np.diag(eigvals[sorted_index]))[:reduced_dimension, :reduced_dimension])
    eigvecs = (eigvecs[:, sorted_index])[:,:reduced_dimension]
    eigvals_sqrt = eigvals ** (1/2)
    
    result = eigvecs @ eigvals_sqrt
        
    return result
    
def ISO(dataset, reduced_dimension, K):
    
    N = dataset.shape[0]
    
    neighbors = NearestNeighbors(K + 1, metric= 'euclidean').fit(dataset)
    
    distance = neighbors.kneighbors_graph(dataset, K + 1).toarray() * squareform(pdist(dataset, metric='euclidean'))
    
    distance[distance == 0] = 10 ** 50
    np.fill_diagonal(distance, 0)
    
    for i in range(0, N):
        for m in range(0, N):
            for h in range(0, N):
                distance[m,h] = min(distance[m,h], distance[m, i] + distance[i, h])
                
    J = np.identity(N) - 1/N * (np.ones((N,1)) @ np.ones((1,N)))
    
    B = -1/2 * J @ np.square(distance) @ J
    
    eigvals, eigvecs = np.linalg.eig(B)
    sorted_index = eigvals.argsort()[::-1]
    eigvals = ((np.diag(eigvals[sorted_index]))[:reduced_dimension, :reduced_dimension])
    eigvecs = (eigvecs[:, sorted_index])[:,:reduced_dimension]
    eigvals_sqrt = eigvals ** (1/2)
    
    result = eigvecs @ eigvals_sqrt
    
    return result
    

def assignment06():
    ''' Assignment 6 for Introduction to Machine Learning, Sp 2018
    Classic Multidimensional Scaling (MDS), ISOMAP, Locally Liner Embedding (LLE)
    ''' 

    # Load Swissroll
    swissroll = np.loadtxt('swissroll.txt') # 3-dimensional data set (N=500)
    swissroll2DPCA = MDS(swissroll, 2)
    swissroll2DISO = ISO(swissroll, 2, 10)
    
    # Load Clusters 
    clusters = np.loadtxt('clusters.txt') # 10-dimensional data set (N=600)
    clusters_labels = clusters[:,10]
    clusters = clusters[:,0:10]
    clusters2DPCA = MDS(clusters, 2)
    clusters2DISO = ISO(clusters, 2, 210)
    
    #Load Halfmoons
    halfmoons = np.loadtxt('halfmoons.txt') 
    halfmoons_labels = halfmoons[:,3] 
    halfmoons = halfmoons[:,0:3]
    halfmoons2DPCA = MDS(halfmoons, 2)
    halfmoons2DISO = ISO(halfmoons, 2, 60)
    
    fig = plt.figure(figsize=(15, 4))
    ax = fig.add_subplot(131, projection='3d')
    ax.plot3D(swissroll[:,0],swissroll[:,1], swissroll[:,2], '.b')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.set_title('Swissroll Data Set')
    
    ax = fig.add_subplot(132, projection='3d')
    ax.plot3D(clusters[clusters_labels==1,0],clusters[clusters_labels==1,1], clusters[clusters_labels==1,2], '.r')
    ax.plot3D(clusters[clusters_labels==2,0],clusters[clusters_labels==2,1], clusters[clusters_labels==2,2], '.g')
    ax.plot3D(clusters[clusters_labels==3,0],clusters[clusters_labels==3,1], clusters[clusters_labels==3,2], '.b')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.set_title('Clusters Data Set')
    
    ax = fig.add_subplot(133, projection='3d')
    ax.plot3D(halfmoons[halfmoons_labels==1,0],halfmoons[halfmoons_labels==1,1], halfmoons[halfmoons_labels==1,2], '.r')
    ax.plot3D(halfmoons[halfmoons_labels==2,0],halfmoons[halfmoons_labels==2,1], halfmoons[halfmoons_labels==2,2], '.b')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.set_title('Halfmoons Data Set')
    
    plt.show()
    
    # Implement Classic MDS, ISOMAP and LLE
    
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.plot(swissroll2DPCA[:,0], swissroll2DPCA[:,1], '.r')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title('Swissroll data set')
    ax = fig.add_subplot(122)
    ax.plot(swissroll2DISO[:,0], swissroll2DISO[:,1], '.r')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title('Swissroll data set')
    
    plt.show()

    
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.plot(clusters2DPCA[clusters_labels==1,0], clusters2DPCA[clusters_labels==1,1], '.r')
    ax.plot(clusters2DPCA[clusters_labels==2,0], clusters2DPCA[clusters_labels==2,1], '.g')
    ax.plot(clusters2DPCA[clusters_labels==3,0], clusters2DPCA[clusters_labels==3,1], '.b')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title('Clusters data set')
    ax = fig.add_subplot(122)
    ax.plot(clusters2DISO[clusters_labels==1,0], clusters2DISO[clusters_labels==1,1], '.r')
    ax.plot(clusters2DISO[clusters_labels==2,0], clusters2DISO[clusters_labels==2,1], '.g')
    ax.plot(clusters2DISO[clusters_labels==3,0], clusters2DISO[clusters_labels==3,1], '.b')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title('Clusters data set')
    
    plt.show()
    
    
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.plot(halfmoons2DPCA[halfmoons_labels==1,0], halfmoons2DPCA[halfmoons_labels==1,1], '.r')
    ax.plot(halfmoons2DPCA[halfmoons_labels==2,0], halfmoons2DPCA[halfmoons_labels==2,1], '.g')
    ax.plot(halfmoons2DPCA[halfmoons_labels==3,0], halfmoons2DPCA[halfmoons_labels==3,1], '.b')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title('Halfmoons data set')
    ax = fig.add_subplot(122)
    ax.plot(halfmoons2DISO[halfmoons_labels==1,0], halfmoons2DISO[halfmoons_labels==1,1], '.r')
    ax.plot(halfmoons2DISO[halfmoons_labels==2,0], halfmoons2DISO[halfmoons_labels==2,1], '.g')
    ax.plot(halfmoons2DISO[halfmoons_labels==3,0], halfmoons2DISO[halfmoons_labels==3,1], '.b')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title('Halfmoons data set')
    
    plt.show()

if __name__ == '__main__':
    
    assignment06()
