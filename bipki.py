import numpy as np
import time

N = int(input())
A = np.zeros((N, N))
b = np.zeros((N))
tmp = [float(i) for i in input().split()]

k=0
for i in range(N):
    for j in range(N):
        A[i,j]=tmp[k]
        k+=1

for i in range(N):
    b[i]=tmp[k]
    k+=1


start = time.time()
x = np.linalg.solve(A, b)
end = time.time()

print("py",N,(end - start)*1000)
