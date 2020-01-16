import matplotlib.pyplot as plt 
import numpy as np    

def plot(w, b):

    s = -w[0]/w[1]
    i = -b/w[1]
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = i + s * x_vals
    plt.plot(x_vals, y_vals, '--')
    plt.xlabel('saturation disparity')
    plt.ylabel('u component range')
    plt.title('decision boundary')

    #fig.savefig('decisionBoundary.png')
    plt.show()

def main():

    w =  [1092, -2106]
    b = 1014.5915523622905
    plot(w, b)

if __name__ == '__main__':
    main()
