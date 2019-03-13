import numpy as np
import copy as cp

def A(a,d): #input vector a and dimension d. Output the symmetric matrix A.
    n=int(d*(d+1)/2)
    A = np.zeros((d,d))
    s=0
    for i in range (d):
        for j in range(i,d):
            A[i,j]=a[s]
            s+=1
    At = cp.deepcopy(A)
    for i in range(d):
        At[i,i] = 0
    A=A+np.transpose(At)
    return A
    
def init(d,N): #input dimension d and number of points N. returns perturbed points z and corresponding labels
    A=np.eye(d)
    c=np.zeros(d)
    z=np.zeros((N,d))
    labels=np.zeros(N)
    #radius = np.zeros(N)
    for i in range (N):
        for j in range (d):
            z[i,j] = np.random.uniform(-1,1) #generate random points
        r=np.matmul(A,z[i]-c)
        r=np.dot(z[i]-c,r)
        if (r <= 1):
            labels[i]=1 #label points inside ellipse
        for j in range (d):
            z [i,j] +=np.random.uniform(-0.5,0.5) #perturb
        #radius[i]=np.sqrt(z[i,0]**2+z[i,1]**2)
    return z,labels

print(init(2,10))
 
 