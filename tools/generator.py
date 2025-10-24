import sys
import numpy as np

def main():
    
    size = int(sys.argv[1])
    
    matrix = np.random.rand(size, size)*100
    vector = np.random.rand(size)*100
    
    print(size)
    for i in range(size):
        for j in range(size):
            print(matrix[i,j],end=" ")

    for j in range(size):
        print(vector[j],end=" ")
if __name__ == "__main__":
    main()
