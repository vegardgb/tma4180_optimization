import numpy as np
import copy as cp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 

def vecToMat(a): #input vector a and dimension 2. Output the symmetric matrix A.
    A = np.zeros((2,2))
    s=0
    for i in range (2):
        for j in range(i,2):
            A[i,j]=a[s]
            s+=1
    At = cp.deepcopy(A)
    for i in range(2):
        At[i,i] = 0
    A=A+np.transpose(At)
    return A

def init(N, x): #

    a = x[:3]
    c = x[3:]
    A = vecToMat(a)
    
    z=np.zeros((N,2))
    labels=np.zeros(N)
    for i in range (N):
        for j in range (2):
            z[i,j] = np.random.uniform(-3,3) #generate random points
        r=np.matmul(A,z[i]-c)
        r=np.dot(z[i]-c,r)
        if (r <= 1):
            labels[i]=1 #label points inside ellipse
        for j in range (2):
            z [i,j] +=np.random.uniform(-0.5,0.5) #perturb
    
    return z,labels

# f2 model

def r_tilde_i(x,zi):
    a = x[:3]
    b = x[3:]
    #the equivalent of matrix multiplication, for saving time, (z-c)^T*A*(z-c) -1
    r1 = a[0]*zi[0]**2+2*a[1]*zi[0]*zi[1]+a[2]*zi[1]**2
    r2 = zi[0]*b[0]+zi[1]*b[1]
    return r1 - r2 - 1

def gradient_r_tilde_i(x,zi):
    gradient=np.zeros(5)
    gradient[0]= zi[0]**2
    gradient[1]= 2*zi[0]*zi[1]
    gradient[2]= zi[1]**2
    gradient[3]= -zi[0]
    gradient[4]= -zi[1]
    
    return gradient    

def f2(x,N,z,labels):
    value = 0
    for i in range (N):
        if (labels[i]==1):
            value += max(r_tilde_i(x,z[i]),0)**2
        else:
            value += min(r_tilde_i(x,z[i]),0)**2
    return value


def gradient_f2(x,N,z,labels):
    gradient=np.zeros(5)
    for i in range (N):
        if (labels[i]==1):
            gradient += (r_tilde_i(x,z[i])+np.abs(r_tilde_i(x,z[i])))*gradient_r_tilde_i(x,z[i])
        else:
            gradient += (r_tilde_i(x,z[i])-np.abs(r_tilde_i(x,z[i])))*gradient_r_tilde_i(x,z[i])
    return gradient

#####
    
# f1 model 

def ri(x,zi):
    a = x[:3]
    c = x[3:]
    #the equivalent of matrix multiplication, for saving time, (z-c)^T*A*(z-c) -1
    r1 = (zi[0]-c[0])*(a[0]*(zi[0]-c[0])+ a[1]*(zi[1]-c[1]))
    r2 = (zi[1]-c[1])*(a[1]*(zi[0]-c[0]) + a[2]*(zi[1]-c[1]))
    return  r1 + r2 - 1

def gradient_ri (x,zi):
    a = x[:3]
    c = x[3:]
    gradient=np.zeros(5)
    gradient[0]=(zi[0]-c[0])**2
    gradient[1]=2*(zi[0]-c[0])*(zi[1]-c[1])
    gradient[2]=(zi[1]-c[1])**2
    gradient[3]=-2*a[0]*(zi[0]-c[0])-2*a[1]*(zi[1]-c[1])
    gradient[4]=-2*a[2]*(zi[1]-c[1])-2*a[1]*(zi[0]-c[0])
    return gradient
    
def f1(x,N,z,labels):
    returnf = 0
    for i in range (N):
        if (labels[i]==1):
            returnf += max(ri(x,z[i]),0)**2
        else:
            returnf += min(ri(x,z[i]),0)**2
    return returnf
    
def gradient_f1(x,N,z,labels):
    gradient=np.zeros(5)
    for i in range (N):
        if (labels[i]==1):
            gradient += (ri(x,z[i])+np.abs(ri(x,z[i])))*gradient_ri(x,z[i])
        else:
            gradient += (ri(x,z[i])-np.abs(ri(x,z[i])))*gradient_ri(x,z[i])
    return gradient

def gradientDescent(N,z,labels, TOL, f, gradient_f):
    #initialization
    x = np.zeros(5)
    #A[0,0]
    x[0]= 1
    #A[0,1]
    x[1]= 0
    #A[1,1]
    x[2]= 1
    
    #armijo parameters
    rho = 0.5
    mu = 0.9
    f_array = []
    i = 0
    while np.linalg.norm(gradient_f(x,N,z,labels)) > TOL and i < 1000:
        p_k = -gradient_f(x,N,z,labels)    
        #armirjo condition for alpha
        alpha = 0.1
        while f((x + alpha*p_k),N,z,labels) > f(x,N,z,labels) + alpha*mu*np.dot(gradient_f(x,N,z,labels),p_k):
            alpha = rho*alpha
        #iteration printer
        if i % 50 == 0:
            print('iteration ', i)
        x += alpha*p_k
        f_array.append(f(x,N,z,labels))
        i = i + 1
    print('Tolerance reached after ', i, ' iterations.')
    
    
    return x, f_array 

#Fletcher-Reeves version of the conjugate gradient method
def conjugateGradient(N, z, labels, c1, c2, TOL, f, gradient_f):
    #initialization
    x = np.zeros(5)
    #A[0,0]
    x[0]= 1
    #A[0,1]
    x[1]= 0
    #A[1,1]
    x[2]= 1
    
    def strong_wolfe_condition(x,alpha,p):
        cond1 = f(x + alpha*p,N,z,labels) <= f(x,N,z,labels) + c1*alpha*np.dot(gradient_f(x,N,z,labels),p)
        cond2 = np.abs(np.dot(gradient_f(x+alpha*p,N,z,labels),p)) <= c2*np.abs(np.dot(gradient_f(x,N,z,labels),p))
        return cond1 , cond2
        
    #initial values
    residual = gradient_f(x,N,z,labels)
    p = - residual
    k = 0
    GDcounter = 0
    breakcounter = 0
    beta = 0
    nextresidual = np.zeros(5)
    
    #main loop
    #terminate = False
    f_array = []
    while np.linalg.norm(gradient_f(x,N,z,labels),2) > TOL and k < 1000:
        
        #print('<gradient,p> = ',np.dot(residual,p))
        
        #checking that p is actually a descent direction, if not, do a gradient descent step
        if np.dot(residual,p) > 0:
            #armirjo condition for alpha
            rho = 0.5
            mu = 0.9
            alpha = 1
            while f((x + alpha*p),N,z,labels) > f(x,N,z,labels) + alpha*mu*np.dot(gradient_f(x,N,z,labels),p):
                alpha = rho*alpha
            p = - residual
            x += alpha*p
            k = k+1
            f_array.append(f(x,N,z,labels))
            GDcounter += 1
            continue
        
        #goldsein/wolfe initial conditions
        alpha = 1
        alpha_min = 0
        alpha_max = np.inf 
        cond1, cond2 = strong_wolfe_condition(x,alpha,p)
        counter = -1
        
        #goldstein/wolfe loop
        while cond1 == False or cond2 ==False:
            counter += 1
            #if counter % 10 == 0:
            #    print('Inner loop iteration', counter)
            #    print('alpha = ', alpha)
            
            #print(cond1,cond2)
            if cond1 == False:
                alpha_max = alpha
                alpha = (alpha_min + alpha_max)/2
            else:
                alpha_min = alpha
                if alpha_max == np.inf:
                    alpha = 2*alpha
                else:
                    alpha = (alpha_min + alpha_max)/2
            if alpha_max - alpha_min < 10**-3:
                breakcounter += 1
                rho = 0.5
                mu = 0.9
                alpha = 1
                while f((x + alpha*p),N,z,labels) > f(x,N,z,labels) + alpha*mu*np.dot(gradient_f(x,N,z,labels),p):
                    alpha = rho*alpha
                p = - residual
                x += alpha*p
                k = k+1
                f_array.append(f(x,N,z,labels))
                GDcounter += 1
                break
            
            cond1, cond2 = strong_wolfe_condition(x,alpha,p)
            
        x += alpha*p
        nextresidual = gradient_f(x,N,z,labels)
        beta = np.dot(nextresidual,nextresidual - residual)/np.dot(residual,residual)
        residual = nextresidual
        p = -residual + beta*p
        k = k+1
        f_array.append(f(x,N,z,labels))
        if k % 20 == 0:
            print('Outer Loop Iteration ', k)
    
    f_array = np.asarray(f_array)
    
    print('finished after ', k , ' iterations.', GDcounter, ' iterations was done with regular Gradient Descent')
    print('Norm of final gradient: ', np.linalg.norm(gradient_f(x,N,z,labels),2))
    print('Inner loop broke ', breakcounter, ' times...')
    
    return x , f_array                

def ellipse_creator(x,exact = False):
    a = x[:3]
    c = x[3:]
    A = vecToMat(a)
    xCenter = c[0]
    yCenter = c[1]
    eigenvals, eigenvecs =np.linalg.eig(A)  
    width = 2/np.sqrt(eigenvals[0])
    height = 2/np.sqrt(eigenvals[1])
    theta = np.arctan2(eigenvecs[1,0],eigenvecs[0,0])*180/np.pi
    if exact:
        ellipse = Ellipse((xCenter,yCenter),width,height,theta, fill=False,linewidth = 2.0,linestyle = '--')
    else:
        ellipse = Ellipse((xCenter,yCenter),width,height,theta, fill=False,linewidth = 2.0)
    return ellipse

def gradient_descent_analysis(N, labels, z, zInterior, zExterior, x_exact):
    #TOL is how small we want the norm of the gradient to be before terminating the loop.
    TOL = 0.5*10**(-1)
    # Gradient descent start at A as Identity matrix and c in origo
       
    x, f1_array = gradientDescent(N,z,labels,TOL, f1, gradient_f1)

    print('function value f1 after tolerance ', TOL, ' reached: ', f1(x,N,z,labels))
    
    approx_ellipse = ellipse_creator(x)
    exact_ellipse = ellipse_creator(x_exact,exact = True)
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)
    ax2.set_title('Gradient Descent with Armijo Conditions -  f1 Model ', fontsize=18)
    ax2.add_patch(approx_ellipse)
    ax2.add_patch(exact_ellipse)
    ax2.scatter(zInterior[:,0],zInterior[:,1])
    ax2.scatter(zExterior[:,0],zExterior[:,1])
    
    plt.show()
    
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(1,1,1)
    ax3.set_title('Convergence of Gradient Descent with Armijo Conditions -  f1 Model ', fontsize=18)
    ax3.plot(f1_array)
    ax3.set_xlabel('Iterations', fontsize=16)
    ax3.set_ylabel('Function value', fontsize=16)
    
    plt.show()

    x, f2_array = gradientDescent(N,z,labels,TOL, f2, gradient_f2)
    print('function value f1 after tolerance ', TOL, ' reached: ', f2(x,N,z,labels))
    
    #converting matrix and vector so that it can represent the ellipse
    AStar = vecToMat(x[:3])
    b = x[3:]
    AStarInv = np.linalg.inv(AStar)
    c = 0.5*np.matmul(AStarInv,b)
    A = AStar*(1-np.linalg.norm(c,2)**2)
    x[0] = A[0,0]
    x[1] = A[0,1]
    x[2] = A[1,1]
    x[3:] = c
    
    approx_ellipse = ellipse_creator(x)
    exact_ellipse = ellipse_creator(x_exact,exact = True)
    
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(1,1,1)
    ax4.set_title('Gradient Descent with Armijo Conditions -  f2 Model ', fontsize=18)
    ax4.add_patch(approx_ellipse)
    ax4.add_patch(exact_ellipse)
    ax4.scatter(zInterior[:,0],zInterior[:,1])
    ax4.scatter(zExterior[:,0],zExterior[:,1])
    
    plt.show()
    
    fig5 = plt.figure()
    ax5 = fig5.add_subplot(1,1,1)
    ax5.set_title('Convergence of Gradient Descent with Armijo Conditions -  f2 Model ', fontsize=18)
    ax5.plot(f2_array)
    ax5.set_xlabel('Iterations', fontsize=16)
    ax5.set_ylabel('Function value', fontsize=16)
    
    plt.show()
    return 0

def conj_gradient_analysis(N, labels, z, zInterior, zExterior,x_exact):
    #TOL is how small we want the norm of the gradient to be before terminating the loop.
    TOL = 0.5*10**(-1)
    
    x, f1_array = conjugateGradient( N, z, labels, 0.01, 0.49, TOL, f1, gradient_f1)    
    
    print('function value f1 when tolerance is reached f1 model ', f1(x,N,z,labels))
    
    approx_ellipse = ellipse_creator(x)
    exact_ellipse = ellipse_creator(x_exact,exact = True)
    
    fig6 = plt.figure()
    ax6 = fig6.add_subplot(1,1,1)
    ax6.set_title('Fletcher-Reeves method -  f1 model', fontsize=18)
    ax6.add_patch(approx_ellipse)
    ax6.add_patch(exact_ellipse)
    ax6.scatter(zInterior[:,0],zInterior[:,1])
    ax6.scatter(zExterior[:,0],zExterior[:,1])
    
    plt.show()
    
    fig7 = plt.figure()
    ax7 = fig7.add_subplot(1,1,1)
    ax7.set_title('Convergence of Fletcher-Reeves method -  f1 Model ', fontsize=18)
    ax7.set_xlabel('Iterations', fontsize=16)
    ax7.set_ylabel('Function value', fontsize=16)
    ax7.plot(f1_array)
    
    plt.show()

    x, f2_array = conjugateGradient( N, z, labels, 10**-6, 0.4, TOL, f2, gradient_f2)    
    print('function value f2 when tolerance is reached f2 model: ', f2(x,N,z,labels))
    #converting matrix and vector so that it can represent the ellipse
    AStar = vecToMat(x[:3])
    b = x[3:]
    AStarInv = np.linalg.inv(AStar)
    c = 0.5*np.matmul(AStarInv,b)
    A = AStar*(1-np.linalg.norm(c,2)**2)
    x[0] = A[0,0]
    x[1] = A[0,1]
    x[2] = A[1,1]
    x[3:] = c
    
    approx_ellipse = ellipse_creator(x)
    exact_ellipse = ellipse_creator(x_exact,exact = True)
    
    fig8 = plt.figure()
    ax8 = fig8.add_subplot(1,1,1)
    ax8.set_title('Fletcher-Reeves method - f2 Model', fontsize=18)
    ax8.add_patch(approx_ellipse)
    ax8.add_patch(exact_ellipse)
    ax8.scatter(zInterior[:,0],zInterior[:,1])
    ax8.scatter(zExterior[:,0],zExterior[:,1])
    
    plt.show()
    
    fig9 = plt.figure()
    ax9 = fig9.add_subplot(1,1,1)
    ax9.set_title('Convergence of Fletcher-Reeves method -  f2 Model ', fontsize=18)
    ax9.plot(f2_array)
    ax9.set_xlabel('Iterations', fontsize=16)
    ax9.set_ylabel('Function value', fontsize=16)
    
    plt.show()    

gamma1= 0.3
gamma2 = 3
beta = 0.5

def c1(x):
    return gamma2-x[0]
def c2(x):
    return x[0]-gamma1
def c3(x):
    return gamma2-x[2]
def c4(x):
    return x[2]-gamma1
def c5(x):
    return np.sqrt(x[0]*x[2])-np.sqrt(gamma1**2+x[1]**2)

def constraints(x):
    const = np.zeros(5)
    const[0] = c1(x)
    const[1] = c2(x)
    const[2] = c3(x)
    const[3] = c4(x)
    const[4] = c5(x)
    return const

def sfunc(x):
    const = constraints(x)
    s = np.zeros(5)
    for i in range (5):
        if const[i]<0:
            return -np.ones(5)
        else:
            s[i]=const[i]
    return s

def B(x,N,z,labels, f):
    s = sfunc(x)
    
    logsum=0
    for i in range(5):
        logsum+= np.log(s[i])
    func = f(x,N,z,labels)
    return func - beta*logsum

def gradientB(x,N,z,labels, gradient_f):
    s = sfunc(x)
    grad_s = np.zeros(5)
    grad_s[0] = - 1/s[0] + 1/s[1] + (0.5/s[4])*np.sqrt(x[2]/x[0])
    grad_s[1] = (x[1]/s[4])/np.sqrt(gamma1**2+x[1]**2) 
    grad_s[2] = - 1/s[2] + 1/s[3] + (0.5/s[4])*np.sqrt(x[0]/x[2])
    grad_f = gradient_f(x,N,z,labels)
    return grad_f  - beta*grad_s

def constrainedGD(TOL, N, labels, z, zInterior, zExterior,f,gradient_f):
    #TOL = 0.5*10**(1)
    x = np.zeros(5)
    #A[0,0]
    x[0]= 1
    #A[0,1]
    x[1]= 0
    #A[1,1]
    x[2]= 1
    
    i = 0
    mu = 0.9
    rho = 0.5
    B_array = []
    alpha= 1
    while np.linalg.norm(gradient_f(x,N,z,labels),2) > TOL and i < 1000:
        p_k = -gradientB(x,N,z,labels,gradient_f)    
        #armirjo condition for alpha, also checks positive definitness of the matrix
        while B((x + alpha*p_k),N,z,labels,f) > B(x,N,z,labels,f) + alpha*mu*np.dot(gradientB(x,N,z,labels,gradient_f),p_k) or sfunc(x + alpha*p_k)[0] == -1:
            alpha = rho*alpha
        xNew = x + alpha*p_k
        x = xNew
            
        
        B_array.append(B(x,N,z,labels,f))
        i = i + 1
        if i % 50 == 0:
            print('iteration', i)
    B_array = np.asarray(B_array)
    print('finished after ',i, ' iterations.')
    return x, B_array

def constrainedGD_analysis(N, labels, z, zInterior, zExterior,x_exact):
    
    TOL = 0.5*10**(-1)
    
    x, B_array = constrainedGD(TOL,N, labels, z, zInterior, zExterior,f1,gradient_f1)
    print('final f1 value: ', f1(x,N,z,labels))
    
    approx_ellipse = ellipse_creator(x)
    exact_ellipse = ellipse_creator(x_exact,exact=True)
    
    fig6 = plt.figure()
    ax6 = fig6.add_subplot(1,1,1)
    ax6.set_title('Constrained Gradient Descent method -  f1 model', fontsize=18)
    ax6.add_patch(approx_ellipse)
    ax6.add_patch(exact_ellipse)
    ax6.scatter(zInterior[:,0],zInterior[:,1])
    ax6.scatter(zExterior[:,0],zExterior[:,1])
    
    plt.show()
    
    fig7 = plt.figure()
    ax7 = fig7.add_subplot(1,1,1)
    ax7.set_title('Convergence of the constrained Gradient Descent method -  f1 Model ', fontsize=18)
    ax7.plot(B_array)
    ax7.set_xlabel('Iterations', fontsize=16)
    ax7.set_ylabel('Function value', fontsize=16)
    
    plt.show()

    x, B_array = constrainedGD(TOL,N, labels, z, zInterior, zExterior,f2,gradient_f2)
    print('final f2 value: ', f2(x,N,z,labels))
    #converting matrix and vector so that it can represent the ellipse
    AStar = vecToMat(x[:3])
    b = x[3:]
    AStarInv = np.linalg.inv(AStar)
    c = 0.5*np.matmul(AStarInv,b)
    A = AStar*(1-np.linalg.norm(c,2)**2)
    x[0] = A[0,0]
    x[1] = A[0,1]
    x[2] = A[1,1]
    x[3:] = c
    
    approx_ellipse = ellipse_creator(x)
    exact_ellipse = ellipse_creator(x_exact,exact=True)
    
    fig6 = plt.figure()
    ax6 = fig6.add_subplot(1,1,1)
    ax6.set_title('Constrained Gradient Descent method -  f2 model', fontsize=18)
    ax6.add_patch(approx_ellipse)
    ax6.add_patch(exact_ellipse)
    ax6.scatter(zInterior[:,0],zInterior[:,1])
    ax6.scatter(zExterior[:,0],zExterior[:,1])
    
    plt.show()
    
    fig7 = plt.figure()
    ax7 = fig7.add_subplot(1,1,1)
    ax7.set_title('Convergence of the constrained Gradient Descent method -  f2 Model ', fontsize=18)
    ax7.plot(B_array)
    ax7.set_xlabel('Iterations', fontsize=16)
    ax7.set_ylabel('Function value', fontsize=16)
    
    plt.show()        

def main():
    
    N = 500  # Number of points

    # Initializing matrix A as diagonal matrix with d dimensions
    #(a[0],a[1},a[2]) = (A[0,1], A[0,0], A[1,1])
    a = np.zeros(3)
    a[0]= 1/3
    a[1]= 1/5
    a[2]= 1/4
    
    # initializing the translation vector to the centrum of ellipse. 
    c=np.asarray((0,0))
    
    x_exact= np.zeros(5)
    x_exact[:3] = a
    x_exact[3:] = c
    
    exact_ellipse = ellipse_creator(x_exact,exact = True)
    
    # Initialize points and labels of two classes
    z,labels =init(N,x_exact)
    
    optimalf1 = f1(x_exact,N,z,labels)
    
    print('"optimal" function value f1/f2', optimalf1)
    
    # Defines two sorted lists of these classes.
    zInterior = z[labels==1]
    zExterior = z[labels==0]
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,1,1)
    ax1.set_title('"Exact" solution', fontsize=18)
    ax1.add_patch(exact_ellipse)
    ax1.scatter(zInterior[:,0],zInterior[:,1])
    ax1.scatter(zExterior[:,0],zExterior[:,1])
    
    plt.show()
    
    
    #gradient_descent_analysis(N, labels, z, zInterior, zExterior,x_exact)
    #conj_gradient_analysis(N, labels, z, zInterior, zExterior,x_exact)
    constrainedGD_analysis(N, labels, z, zInterior, zExterior,x_exact)

main()