"""
Data generation for a set of predefined functions. Generates both testing and training data for set of test functions. 



"""


from math import sqrt
import os
import numpy as np


root_folder = os.path.dirname(os.path.abspath(__file__))

## Example 1: Random walk changes color when y < 0.5

def genRandomWalkPoints(B,T, rng, outfun):    
    X = np.zeros((B, T, 2))
    X[:,0,:] = rng.randn(B, 2) - 0.5
    for t in range(1,T):
        step = rng.randn(B, 2) * 0.1
        X[:,t,:] = X[:,t-1,:] + step
    X = X.astype(np.float32)
    
    Y = np.zeros((B, T), dtype=int)
    for i in range(B):
        for j in range(T):
            Y[i,j] = outfun(X[i,j])
    Y = Y.astype(np.float32)            
    return X, Y
    
def thresholdFun(coord):
    x,y = coord[:]
    if(y < 0.5):
        return 1
    else:
        return 0



rng = np.random.RandomState(10)
X_train, Y_train = genRandomWalkPoints(100, 10,rng, thresholdFun)
X_test, Y_test = genRandomWalkPoints(30, 10, rng, thresholdFun)


os.makedirs(f'{root_folder}/data/classification_example/', exist_ok=True)
np.save(f'{root_folder}/data/classification_example/train_ex_data.npy', X_train)
np.save(f'{root_folder}/data/classification_example/train_ex_labels.npy', Y_train)

np.save(f'{root_folder}/data/classification_example/test_ex_data.npy', X_test)
np.save(f'{root_folder}/data/classification_example/test_ex_labels.npy', Y_test)

print("Done with classification")

## Example 3: Hall of mirrors. state = (x,y, vx, vy)
# A set of planes (m_i, r_i) Every time ((x,y) dot m) > m

def genTransitionPoints(B,T, init,  outfun):
    X = np.zeros((B, T, 4))
    X[:,0,:] = init
    Y = np.zeros((B, T, 4))
    for t in range(1,T+1):
        for i in range(B):
            tmp = np.array( outfun(X[i, t-1]) )
            if(i==0): print (str(t)+","+ str(i) + ": "+ str(X[i,t-1]) + "->" + str(tmp))
            Y[i,t-1] = tmp
            if(t < T):
                X[i,t] = tmp
    X = X.astype(np.float32)    
    Y = Y.astype(np.float32)    
    return X, Y 

def mirrors(point):
    x,y,vx,vy = point[:]
    wall1 = (1, 0.5, 3)
    wall2 = (-1, 0.5, 3)
    wall3 = (0, 1, 3)
    wall4 = (0, -1, 3)
    def checkwall(wall, x,y, vx, vy):
        return x*wall[0] + y*wall[1] > wall[2] and vx*wall[0] + vy*wall[1] > 0
    def reflect(wall, x, y, vx, vy):
        wnorm = sqrt(wall[0]**2 + wall[1]**2)
        dot = (vx*wall[0] + vy*wall[1])/wnorm
        return x, y, vx - 2*(wall[0]/wnorm)*dot , vy - 2*(wall[1]/wnorm)*dot
    
    if checkwall(wall1, x,y,vx, vy):
        return reflect(wall1, x, y, vx, vy)
    if checkwall(wall2, x,y,vx, vy):
        return reflect(wall2, x, y, vx, vy)
    if checkwall(wall3, x,y,vx, vy):
        return reflect(wall3, x, y, vx, vy)
    if checkwall(wall4, x,y,vx, vy):
        return reflect(wall4, x, y, vx, vy)
    return x+vx,y+vy,vx,vy

B = 100
X_train, Y_train = genTransitionPoints(B, 20, (rng.randn(B, 4) - 0.5)*0.3, mirrors)
B = 30
X_test, Y_test = genTransitionPoints(B, 20, (rng.randn(B, 4) - 0.5)*0.3, mirrors)



os.makedirs(f'{root_folder}/data/mirrors_example/', exist_ok=True)
np.save(f'{root_folder}/data/mirrors_example/train_ex_data.npy', X_train)
np.save(f'{root_folder}/data/mirrors_example/train_ex_labels.npy', Y_train)

np.save(f'{root_folder}/data/mirrors_example/test_ex_data.npy', X_test)
np.save(f'{root_folder}/data/mirrors_example/test_ex_labels.npy', Y_test)

print("Done with mirrors")

## Example 3: A ball with state (x, y, vx, vy) moves according to the policy: 
## vx' = ite(x > 10 && vx > 0): -vx else vx
## vy' = ite(y < 0 && vy < 0): -vy else vy - 0.98
## x' = ite(x > 10 && vx > 0): x-vx else x + vx
## y' = ite(y < 0 && vy < 0): y-vy else y + vy

def floorBounce(point):
    x,y,vx,vy = point[:]
    if (x > 5 and vx > 0):  
        vxp =  -vx 
        xp = x - vx*0.1
    else: 
        vxp = vx
        xp = x + vx*0.1
    if(y<0 and vy < 0):
        vyp = -vy*0.8
        yp = y - vy*0.1
    else:
        vyp = vy - 0.98
        yp = y + vy*0.1
    return xp, yp, vxp, vyp
        
B = 500
init = (rng.randn(B, 4) - 0.5)*0.4 + np.array([0, 5.0, 3.5, 0])
X_train, Y_train = genTransitionPoints(B, 35, init, floorBounce)
B = 60
init = (rng.randn(B, 4) - 0.5)*0.4 + np.array([0, 5.0, 3.5, 0])
X_test, Y_test = genTransitionPoints(B, 35, init, floorBounce)



os.makedirs(f'{root_folder}/data/bounce_example/', exist_ok=True)
np.save(f'{root_folder}/data/bounce_example/train_ex_data.npy', X_train)
np.save(f'{root_folder}/data/bounce_example/train_ex_labels.npy', Y_train)

np.save(f'{root_folder}/data/bounce_example/test_ex_data.npy', X_test)
np.save(f'{root_folder}/data/bounce_example/test_ex_labels.npy', Y_test)

print("Done with bounce")