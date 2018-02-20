def assignment07():

	''' Assignment 07 for Introduction to Machine Learning, Sp 2018
		EM Algorithm and Gaussian Mixture Models
	''' 
    
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from numpy.matlib import repmat

# Load Data
X = np.loadtxt('GaussianMixture.txt')
fig = plt.figure()
plt.plot(X[:,0],X[:,1],'.b')

def EM_GaussianMixture(X, NumComponents):
    MaximumNumberOfIterations = 600
    DiffThresh = 1e-4
    N, D = X.shape
    
    # Initialize Parameters of each Component K
    Means = np.zeros((NumComponents,D))
    Sigs = np.zeros(((D, D, NumComponents)))
    Ps = np.zeros(NumComponents)
    for i in range(NumComponents):
        rVal = np.random.uniform(0,1)
        Means[i,:] = X[max(1,round(N*rVal)),:]#why use the max function here?
        Sigs[:,:,i] = 1*np.eye(D) #why multiply by 1?
        Ps[i] = 1/NumComponents
        
    # E-Step Solve for p(z | x, Theta(t))
    pZ_X = np.zeros((N,NumComponents))
    for k in range(NumComponents):
        # Assign each point to a Gaussian component with probability pi(k)
        pZ_X[:,k] = multivariate_normal.pdf(X, Means[k,:], Sigs[:,:,k])
    
    pZ_X /= np.sum(pZ_X, axis = 1)[:, None]
    
    Diff = np.inf
    NumberIterations = 1
    while Diff > DiffThresh and NumberIterations <= MaximumNumberOfIterations:
        ## M-step: Update Means, Sigs, Ps
        MeansOld = np.array(Means)
        SigsOld = np.array(Sigs)
        PsOld = np.array(Ps)
        for k in range(NumComponents):
            # Complete M-step: Update parameters
            Means[k,:] = pZ_X[:,k].T@X/(np.sum(pZ_X[:,k]))
            Sigs[:,:,k] = pZ_X[:,k].T@(X - Means[k,:])**2/(np.sum(pZ_X[:,k])) * np.eye(D)
            Ps = np.sum(pZ_X, axis=0)/N
            
        ## E-step: Solve for p(z | x, Theta(t))
        for k in range(NumComponents):
            pZ_X[:,k] = multivariate_normal.pdf(X, Means[k,:], Sigs[:,:,k])
        pZ_X /= np.sum(pZ_X, axis = 1)[:, None]

        # Complete E-step
        
        Diff = sum(sum(abs(MeansOld - Means))) + sum(sum(sum(abs(SigsOld - Sigs)))) + sum(abs(PsOld - Ps))
        print(NumberIterations)
        NumberIterations = NumberIterations + 1
    return Means, Sigs, Ps, pZ_X

# Set number of componenets
NumComponents = 3
EM_Means, EM_Sigs, EM_Ps, pZ_X = EM_GaussianMixture(X, NumComponents)

print('----------------------------')
print('----------------------------')
print('EM Algorithm')
print('# Components: ' + str(NumComponents))
print('----------------------------')
print('Estimated Means')
print(EM_Means)
for i in range(NumComponents):
    print('Estimated Covariances')
    print(EM_Sigs[:,:,i])
    print('Estimated Weights')
    print(EM_Ps)
    
fig = plt.figure(figsize=(15, 4))
plt.suptitle('EM Algorithm')
for i in range(NumComponents):
    ax = fig.add_subplot(1,NumComponents,i+1)
    p1 = ax.scatter(X[:,0], X[:,1], c=pZ_X[:,i]) 
    ax.set_title('Mean: '+ str(EM_Means[i,:]))
    fig.colorbar(p1, ax=ax)

if __name__ == '__main__':
    assignment07()

