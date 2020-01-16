import cv2
import glob
import numpy as np
import svm
import matplotlib.pyplot as plt
import imgProc
import clustering
import lines


imageI = [cv2.imread(file) for file in glob.glob("C:\\Users\\Gautham\\Documents\\PML\\cell_images\\PMLProject\\CreativeComponent\\Dataset\\Infected\\*.png")]

S = []
D = []
C = []
m = 24
e = 3.57
i = 0
for img in imageI:
    print(i)
    i += 1
    h, s, v = imgProc.colorConvert1(img)
    thresh = imgProc.threshold(s,50)
    whites = np.sum(thresh == 255)
    threshClustering = cv2.resize(thresh, (50, 50))
    p = np.argwhere(threshClustering == 255)
    p = p.tolist()
    C.append(clustering.dbscan(m, e, p, img))
    y, u, v = imgProc.colorConvert2(img)

    if np.amax(u) - np.amin(u) >= 40:
        D.append(0)
    else:
        D.append(1)
    if whites != 0:
        S.append([1])
    else:
        S.append([0])


imageU = [cv2.imread(file) for file in glob.glob("C:\\Users\\Gautham\\Documents\\PML\\cell_images\\PMLProject\\CreativeComponent\\Dataset\\Uninfected\\*.png")]
i = 0
for img in imageU:
    print(i)
    i += 1    
    h, s, v = imgProc.colorConvert1(img)
    thresh = imgProc.threshold(s,50)
    whites = np.sum(thresh == 255)
    threshClustering = cv2.resize(threshClustering, (50, 50))
    p = np.argwhere(threshClustering == 255)
    p = p.tolist()
    C.append(clustering.dbscan(m, e, p, img))
    y, u, v = imgProc.colorConvert2(img)

    if np.amax(u) - np.amin(u) >= 40:
        D.append(0)
    else:
        D.append(1)

    if whites != 0:
        S.append([1])
    else:
        S.append([0])


S = np.asarray(S)
D = np.asarray(D)
C = np.asarray(C)

x = np.column_stack((S, D, C))

l = x.shape[0]
n = int(l/2)
y1t = np.ones((n, 1))
y2t = -1*np.ones((n, 1))
yt = np.concatenate((y1t, y2t), axis = 0)

[w, b] = svm.smo(x, yt)

y1 = np.ones((49, 1))
y2 = -1*np.ones((49, 1))
yp = np.concatenate((y1, y2), axis = 0)


#w = [[1.], [0.]]
#b = -0.43617021276595747
error = 0
p = "C:\\Users\\Gautham\\Documents\\PML\\cell_images\\PMLProject\\CreativeComponent\\Dataset\\test\\img"
for i in range(1, 98):
    path = p + str(i) + ".png"
    img = cv2.imread(path)
    print(i)
    h, s, v = imgProc.colorConvert1(img)
    thresh = imgProc.threshold(s,50)
    whites = np.sum(thresh == 255)
    threshClustering = cv2.resize(thresh, (50, 50))
    p = np.argwhere(threshClustering == 255)
    p = p.tolist()
    c = clustering.dbscan(m, e, p, img)
    y, u, v = imgProc.colorConvert2(img)

    if np.amax(u) - np.amin(u) >= 40:
        td = 0
    else:
        td = 1
    whites = np.sum(thresh == 255)
    if whites != 0:
        ts = 1
    else:
        ts = 0

    t = np.array([ts, td])
    yPredicted = svm.predict(t, w, b)
    
    if yPredicted != yp[i]:
        error += 1
    
print(' SVM Accuracy = ', (1 - (error/98))*100, '%')

