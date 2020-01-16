import numpy as np
import cv2

def randSelect(i, m):

    r = np.random.randint(0, m)
    if r != i:
        return r
    else:
        while (r == i):
            r = np.random.randint(0, m)
        return r
    
def calcError(x, y, w, b):
    return np.sign(np.dot(w.T, x.T) + b).astype(int) - y

def computeLH(ai, aj, C, yi, yj):
    if(yi != yj):
        return (max(0, aj-ai), min(C, C + aj - ai))
    else:
        return (max(0, ai + aj - C), min(C, ai + aj))
        
def computeN(xi, xj):
    return -2*np.dot(xi, xj) + np.dot(xi, xi) + np.dot(xj, xj)

def clipAlpha(a, L, H):

    ap = max(a, L)
    return min(ap, H)

def findBW(a, x, y):

    w = np.dot(x.T, np.multiply(a,y))
    b = y - np.dot(x, w)
    return [w, np.mean(b)]

def smo(x, y):

    m, n = x.shape
    print(m, n)
    maxPasses = 100
    alphas = np.zeros((m, 1))
    bias = 0
    passes = 0
    C = 1
    oldAlphas = np.zeros((m, 1))
    while passes < maxPasses:
        print('passes = ', passes)
        oldAlphas = np.copy(alphas)
        for j in range(0, m):
            i = randSelect(j, m)
            eta = computeN(x[i], x[j])
            if eta == 0:
                continue
            oAi = alphas[i]
            oAj = alphas[j]
            [L, H] = computeLH(oAi, oAj, C, y[i], y[j])
            [w, b] = findBW(alphas, x, y)
            Ei = calcError(x[i], y[i], w, b)   
            Ej = calcError(x[j], y[j], w, b)
            alphas[j] = oAj + (y[j]*(Ei - Ej))/eta
            alphas[j] = clipAlpha(alphas[j], L, H)
            alphas[i] = oAi + y[i]*y[j]*(oAj - alphas[j])
        
        passes += 1
    print('params = ', [w, b])
    return [w, b]

def predict(x, w, b):

    return np.sign(np.dot(x, w) + b).astype(int)
    
def main():
    
    a = 2 + np.random.rand(50,2)
    b = 5 + np.random.rand(50,2)
    x = np.concatenate((a, b), axis=0)

    y1 = -1*np.ones((50, 1))
    y2 = np.ones((50, 1))
    y = np.concatenate((y1, y2), axis = 0)
    [w, b] = smo(x, y)
    t = np.array([2, 2])
    print(predict(t, w, b))

if __name__ == '__main__':
    main()
