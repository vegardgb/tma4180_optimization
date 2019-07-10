import numpy as np
import copy as cp
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def init(N, x):  
    c_radius = x[0]
    d_radius = x[1]
    c_center = x[2:4]
    d_center = x[4:]
    
    z=np.zeros((N,2))
    labels=np.zeros(N)
    c_A = np.eye(2)
    c_A[0,0] = c_A[1,1] = c_radius**2
    d_A = np.eye(2)
    d_A[0,0] = d_A[1,1] = d_radius**2
    for i in range (N):
        for j in range (2):
            z[i,j] = np.random.uniform(-3,3) #generate random points
        if ((z[i,0]-c_center[0])**2 + (z[i,1]-c_center[1])**2 <= c_radius**2):
            labels[i] = 1 #label points inside circe C
        if ((z[i,0]-d_center[0])**2 + (z[i,1]-d_center[1])**2 <= d_radius**2):
            labels[i] = 2 #label points inside circle D      
        for j in range (2):
            z [i,j] +=np.random.uniform(-0.3,0.3) #perturb
    
    return z,labels

def ri(radius,center,zi):
    return np.linalg.norm(zi - center,2)**2 - radius**2
    
def c_gradient_ri(x,zi):
    grad = np.zeros(6)
    grad[0] = -2*x[0]
    grad[2] = -2*(zi[0]-x[2])
    grad[3] = -2*(zi[1]-x[3])
    return grad

def d_gradient_ri(x,zi):
    grad = np.zeros(6)
    grad[1] = -2*x[1]
    grad[4] = -2*(zi[0]-x[4])
    grad[5] = -2*(zi[1]-x[5])  
    return grad

def f3(x,N,z,labels):
    f = 0
    for i in range(N):
        if (labels[i]==1): 
            f += max(ri(x[0],x[2:4],z[i]),0)**2
        if (labels[i] ==2) or (labels[i] == 0):
            f += min(ri(x[0],x[2:4],z[i]),0)**2
        if (labels[i]==2): 
            f += max(ri(x[1],x[4:],z[i]),0)**2
        if (labels[i]==1) or (labels[i] == 0):
            f += min(ri(x[1],x[4:],z[i]),0)**2

    return f
    
def gradient_f3(x,N,z,labels):
    grad = np.zeros(6)
    for i in range(N):
        if (labels[i]==1): 
            grad += (ri(x[0],x[2:4],z[i]) + np.abs(ri(x[0],x[2:4],z[i])))*c_gradient_ri(x,z[i])
        if (labels[i]==2) or (labels[i] == 0):
            grad += (ri(x[0],x[2:4],z[i]) - np.abs(ri(x[0],x[2:4],z[i])))*c_gradient_ri(x,z[i])
        if (labels[i]==2): 
            grad += (ri(x[1],x[4:],z[i]) + np.abs(ri(x[1],x[4:],z[i])))*d_gradient_ri(x,z[i])
        if (labels[i] ==1) or (labels[i] == 0):
            grad += (ri(x[1],x[4:],z[i]) - np.abs(ri(x[1],x[4:],z[i])))*d_gradient_ri(x,z[i])
    return grad

beta = 0.5

def c1(x):
    return x[0]
def c2(x):
    return x[1]
def c3to6(x):
    return np.linalg.norm(x[2:4]-x[4:],2)-x[0]-x[1]

def constraints(x):
    const = np.zeros(6)
    const[0] = c1(x)
    const[1] = c2(x)
    const[2] = c3to6(x)
    const[3] = c3to6(x)
    const[4] = c3to6(x)
    const[5] = c3to6(x)
    return const

def sfunc(x):
    const = constraints(x)
    s = np.zeros(6)
    for i in range (6):
        if const[i]<0:
            return -np.ones(6)
        else:
            s[i]=const[i]
    return s

def B(x,N,z,labels):
    s = sfunc(x) 
    logsum=0
    for i in range(6):
        logsum+= np.log(s[i])
    f = f3(x,N,z,labels)
    return f - beta*logsum

def gradientB(x,N,z,labels):
    s = sfunc(x)
    norm_cd = np.linalg.norm(x[2:4]-x[4:],2)
    grad_s = np.zeros(6)
    grad_s[0] = s[0]
    grad_s[1] = s[1]
    grad_s[2:] = -s[0]-s[1]+s[2]*(x[2]-x[4])/norm_cd +s[3]*(x[3]-x[5])/norm_cd - s[4]*(x[2]-x[4])/norm_cd - s[5]*(x[3]-x[5])/norm_cd
    grad_f = gradient_f3(x,N,z,labels)
    return grad_f  - beta*grad_s

def circle_creator(x,exact = False):
    c_radius = x[0]
    d_radius = x[1]
    c_center = x[2:4]
    d_center = x[4:]
    if exact:
        circle1 = Ellipse((c_center[0],c_center[1]),2*c_radius,2*c_radius,0, fill=False,linewidth = 2.0,linestyle = '--')
        circle2 = Ellipse((d_center[0],d_center[1]),2*d_radius,2*d_radius,0, fill=False,linewidth = 2.0,linestyle = '--')
    else:
        circle1 = Ellipse((c_center[0],c_center[1]),2*c_radius,2*c_radius,0, fill=False,linewidth = 2.0)
        circle2 = Ellipse((d_center[0],d_center[1]),2*d_radius,2*d_radius,0, fill=False,linewidth = 2.0)
    return circle1, circle2

def constrainedGD(N, labels, z):
    #TOL = 0.5*10**(1)
    x = np.zeros(6)
    
    #Circle C initial guesses
    x[0] = 0.5 #radius
    x[2] = 2 #x-component of center
    x[3] = -0.5  #y-component of center
    
    #Circle D initial guesses
    x[1]= 0.5 #radius
    x[4] = -2  #x-component of center
    x[5] = 0.5 #y-component of center
    
    i = 0
    mu = 0.9
    rho = 0.5
    B_array = []
    alpha= 1
    for i in range(1000):
        p_k = -gradientB(x,N,z,labels)    
        #armirjo condition for alpha, also checks positive definitness of the matrix
        while B((x + alpha*p_k),N,z,labels) > B(x,N,z,labels) + alpha*mu*np.dot(gradientB(x,N,z,labels),p_k) or sfunc(x + alpha*p_k)[0] == -1:
            alpha = rho*alpha
        xNew = x + alpha*p_k
        x = xNew
        
        
        B_array.append(B(x,N,z,labels))
        i = i + 1
        if i % 10 == 0:
            print(i)
    B_array = np.asarray(B_array)
    print('finished after ',i, ' iterations.')
    return x, B_array

def main():
    N = 500  # Number of points
    
    x_exact= np.zeros(6)
    x_exact[0] = 1 #radius circle C
    x_exact[1] = 1 #radius circle D
    x_exact[2:4] = [0,1] #sentrum of circle C
    x_exact[4:] = [0,-1] #sentrum of circle D
    
    
    exact_circle1, exact_circle2 = circle_creator(x_exact,exact = True)
    
    # Initialize points and labels of two classes
    z,labels =init(N,x_exact)
    
    # Defines two sorted lists of these classes.
    z1_interior = z[labels==1] #points inside circle 1
    z2_interior = z[labels==2] #points inside circle 2
    z_exterior  = z[labels==0] #points outside both circles

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,1,1)
    ax1.set_title('"Exact" solution', fontsize=18)
    ax1.add_patch(exact_circle1)
    ax1.add_patch(exact_circle2)
    ax1.scatter(z1_interior[:,0],z1_interior[:,1])
    ax1.scatter(z2_interior[:,0],z2_interior[:,1])
    ax1.scatter(z_exterior[:,0],z_exterior[:,1])
    
    plt.show()
    
    x, B_array = constrainedGD(N, labels, z)
    
    approx_circle_c, approx_circle_d = circle_creator(x)
    exact_circle_c, exact_circle_d = circle_creator(x_exact,exact = True)
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)
    ax2.set_title('Constrained gradient descent -  f3 model', fontsize=18)
    ax2.add_patch(approx_circle_c)
    ax2.add_patch(approx_circle_d)
    ax2.add_patch(exact_circle_c)
    ax2.add_patch(exact_circle_d)
    ax2.scatter(z1_interior[:,0],z1_interior[:,1])
    ax2.scatter(z2_interior[:,0],z2_interior[:,1])
    ax2.scatter(z_exterior[:,0],z_exterior[:,1])
    
    plt.show()
    
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(1,1,1)
    ax3.set_title('Convergence of the constrained gradient descent -  f3 Model ', fontsize=18)
    ax3.plot(B_array)
    ax3.set_xlabel('Iterations', fontsize=16)
    ax3.set_ylabel('Function value', fontsize=16)
    
    plt.show()

main()