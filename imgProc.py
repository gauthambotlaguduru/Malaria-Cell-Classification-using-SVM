import cv2
import numpy as np

def bgr2yuv(B, G, R):

    Y =  0.299*R + 0.587*G + 0.114*B
    U = -0.147*R - 0.289*G + 0.436*B
    V =  0.615*R - 0.515*G - 0.100*B
    return [np.uint8(Y), np.uint8(U), np.uint8(V)]

def bgr2hsv(b, g, r):

    bd = b/255
    gd = g/255
    rd = r/255
    cmax = max(bd, gd, rd)
    cmin = min(bd, gd, rd)
    delta = cmax - cmin

    if delta == 0:
        h = 0
    elif cmax == rd:
        h = (60 * ((gd-bd)/delta) + 360) % 360
    elif cmax == bd:
        h = (60 * ((rd-gd)/delta) + 240) % 360
    else:
        h = (60 * ((b-r)/delta) + 120) % 360

    if cmax != 0:
        s = (delta/cmax)*100
    else:
        s = 0

    return [np.uint8(h), np.uint8(s), np.uint8(cmax*100)]


def threshold(img, t):

    [r, c] = img.shape
    for i in range(0, r):
        for j in range(0, c):
            if img[i, j] <= t:
                img[i, j] = np.uint8(0)
            else:
                img[i, j] = np.uint8(255)
    return img

def colorConvert1(image):

    b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    [h, w] = b.shape
    o1, o2, o3 = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
    for i in range(0, h):
        for j in range(0, w):
            [h, s, v] = bgr2hsv(b[i, j], g[i, j], r[i, j])
            #print(h, s, v)
            o1[i][j] = h
            o2[i][j] = s
            o3[i][j] = v
    return [o1, o2, o3]

def colorConvert2(image):

    b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    [h, w] = b.shape
    o4, o5, o6 = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
    for i in range(0, h):
        for j in range(0, w):
            [y, u, v] = bgr2yuv(b[i, j], g[i, j], r[i, j])
            o4[i][j] = y
            o5[i][j] = u
            o6[i][j] = v

    return [o4, o5, o6]

def main():

    image = cv2.imread("C:\\Users\\Gautham\\Documents\\PML\\cell_images\\PMLProject\\CreativeComponent\\Dataset\\Infected\\1.png")
    [h, s, v] = colorConvert1(image)
    thresh = threshold(s, 50)
    cv2.imshow('s', s)
    cv2.waitKey(0)
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y1, u1, v1 = cv2.split(yuv)
    [y, u, v] = colorConvert2(image)
    cv2.imshow('u', u)
    cv2.waitKey(0)
    cv2.imshow('u1', u1)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()

    
