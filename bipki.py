import numpy as np
import time

# Простая версия для быстрого замера
np.random.seed(42)
A = np.random.rand(10000, 10000) + np.eye(10000) * 1000
b = np.random.rand(10000)

print("Замер времени решения СЛАУ 1000x1000...")
start = time.time()
x = np.linalg.solve(A, b)
end = time.time()

print(f"Время выполнения: {end - start:.4f} секунд")
print(f"Невязка: {np.linalg.norm(A @ x - b):.2e}")
