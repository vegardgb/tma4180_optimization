#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 16:01:34 2019

@author: vegardgb
"""

import numpy as np
import copy as cp
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def vecToMat(a,d): #input vector a and dimension d. Output the symmetric matrix A.
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

"""

Input dimension d and number of points N. returns perturbed points z and corresponding labels

Parameters:
    d dimensions
    N points
    A start matrix 
    c translation vector. 
    

"""
    
def init(d, N, A, c): #
    
    z=np.zeros((N,d))
    labels=np.zeros(N)
    #print(np.linalg.eig(A))
    #radius = np.zeros(N)
    for i in range (N):
        for j in range (d):
            z[i,j] = np.random.uniform(-1,1) #generate random points
        r=np.matmul(A,z[i]-c)
        r=np.dot(z[i]-c,r)
        if (r <= 1):
            labels[i]=1 #label points inside ellipse
        for j in range (d):
            z [i,j] +=np.random.uniform(-0.1,0.1) #perturb
        #radius[i]=np.sqrt(z[i,0]**2+z[i,1]**2)
    
    return z,labels

"""

r_i(A,c, z,i)

a = [A11, A12, A22]

"""


def ri(a,c,zi):
    A=vecToMat(a,2)
    value = np.matmul(A,zi-c)
    value = np.dot(zi-c,value)
    return value - 1

def gradient_ri (a,c,zi):
    gradient=np.zeros(5)
    gradient[0]=(zi[0]-c[0])**2
    gradient[1]=2*(zi[0]-c[0])*(zi[1]-c[1])
    gradient[2]=(zi[1]-c[1])**2
    gradient[3]=-2*a[0]*(zi[0]-c[0])-2*a[1]*(zi[1]-c[1])
    gradient[4]=-2*a[2]*(zi[1]-c[1])-2*a[1]*(zi[0]-c[0])
    return gradient
    
    
def gradient_f1(a,c,d,N,z,labels):
    gradient=np.zeros(5)
    for i in range (N):
        if (labels[i]==1):
            gradient = (ri(a,c,z[i])+np.abs(ri(a,c,z[i])))*gradient_ri(a,c,z[i])
        else:
            gradient = (ri(a,c,z[i])-np.abs(ri(a,c,z[i])))*gradient_ri(a,c,z[i])
    return gradient
    
 
def gradientDescent(d,N,z,labels):
    A=np.eye(2)
    a = np.zeros(3)
    a[0]= A[0,0]
    a[1]= A[0,1]
    a[2]= A[1,1]
    c=np.zeros(d)
    x = np.zeros(5)
    x[0:3] = a
    x[3:] = c
    alpha =0.001
    for i in range(1000):
        p_k = -gradient_f1(a,c,d,N,z,labels)
        x += alpha*p_k
    print('For loop has terminated')
    A=vecToMat(x[0:3],d)
    c=x[3:]
    return A,c


def conjugateGradient(d,N,z,labels):
    
    r = np.matmul(A,x) - b
    p = - r
    k = 0
    rInner = np.dot(np.transpose(r),r)
    while (r!= 0):
        Apk =np.matmul(A,p)
        nevner = np.dot(np.transpose(p),Apk)
        alpha = rInner/nevner
        x += aplha*p
        newR = r + alpha*Apk
        rInnerNew = np.dot(np.transpose(newR),newR)
        beta = rInnerNew/rInner
        p = -newR +beta*p
        k += 1
        r = newR
        rInner= rInnerNew



def main():
    
        
    d = 2    # d dimensions
    N = 100  # Iterations during gradient descent

    # Initializing matrix A as diagonal matrix with d dimensions
    A=np.eye(d)
    A[0,0] = 25
    A[0,1] = 0
    A[1,1] = 5
    A[1,0] = 0
    
    # initializing the translation vector to the centrum of ellipse. 
    c=np.zeros(d)
    
    # Initialize points and labels of two classes
    z,labels =init(d,N,A,c)
    
    
    # Defines two sorted lists of these classes.
    zInterior = z[labels==1]
    zExterior = z[labels==0]
    
    # Gradient descent start at A as Identity matrix and c in origo
    A,c = gradientDescent(d,N,z,labels)

    yCenter = c[1]
    xCenter = c[0]
    eigenvals, eigenvecs =np.linalg.eig(A)
    width = 1/np.sqrt(eigenvals[1])
    height = 1/np.sqrt(eigenvals[0])
    theta = np.arctan2(eigenvecs[0][1],eigenvecs[0][0])
    print(eigenvecs[0], eigenvecs[1])
    print(eigenvals)
    print(theta)
    approx_ellipse = Ellipse((xCenter,yCenter),width,height,theta, fill=False)
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,1,1)
    ax1.add_patch(approx_ellipse)
    ax1.scatter(zInterior[:,0],zInterior[:,1])
    ax1.scatter(zExterior[:,0],zExterior[:,1])
    
    plt.show()


main()

 