#ifndef __TOOLS_HPP__
#define __TOOLS_HPP__
#include "itersolver.hpp"
#include "xtensor/core/xtensor_forward.hpp"
#include "xtensor/generators/xbuilder.hpp"
#include "xtensor/views/xslice.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <random>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/views/xview.hpp>

template <class T>
auto Residual(xt::xarray<T> A, xt::xarray<T> x, xt::xarray<T> f) {
  return f - xt::linalg::dot(A, x);
}

template <class T> std::pair<xt::xarray<T>, xt::xarray<T>> read_from_stdin() {
  size_t N;
  std::cin >> N;
  xt::xarray<T> retM = xt::zeros<T>({N, N});
  xt::xarray<T> retV = xt::zeros<T>({N});
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      std::cin >> retM(i, j);
  for (int i = 0; i < N; i++)
    std::cin >> retV(i);
  return {retM, retV};
}

template <class T> auto genbad(xt::xarray<T> A, xt::xarray<T> b) {
  return xt::linalg::solve(A, b) + 0.1;
}

// Генератор случайных чисел
static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_real_distribution<double> dis(0.0, 1.0);

// Генерация диагонально доминирующей матрицы (подходит для Якоби и Зейделя)
template <class T> xt::xarray<T> generate_diagonally_dominant_matrix(int n) {
  xt::xarray<T> A = xt::zeros<T>({n, n});

  for (int i = 0; i < n; ++i) {
    T row_sum = 0.0;
    for (int j = 0; j < n; ++j) {
      if (i != j) {
        A(i, j) = dis(gen) * 0.1; // Малые недиагональные элементы
        row_sum += std::abs(A(i, j));
      }
    }
    A(i, i) = row_sum + 1.0; // Диагональный элемент больше суммы остальных
  }

  return A;
}

// Генерация симметричной положительно определенной матрицы (подходит для
// MINRES)
template <class T> xt::xarray<T> generate_spd_matrix(size_t n) {
  xt::xarray<T> A = xt::zeros<T>({n, n});

  // Генерируем случайную матрицу
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= i; ++j) {
      A(i, j) = dis(gen);
      A(j, i) = A(i, j); // Симметрия
    }
  }

  // Делаем положительно определенной: A = A^T * A + n*I
  xt::xarray<T> At = xt::transpose(A);
  A = xt::linalg::dot(At, A) + n * xt::eye<T>({n, n});

  return A;
}

// Генерация матрицы для fixed_point_solver
template <class T> xt::xarray<T> generate_fixed_point_matrix(int n) {
  // Для fixed_point_solver нужна матрица с малым спектральным радиусом
  xt::xarray<T> A = xt::zeros<T>({n, n});

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      A(i, j) = dis(gen) * 0.1 / n; // Очень малые элементы
    }
    A(i, i) += 0.5; // Увеличиваем диагональ для сходимости
  }

  return A;
}

// Генерация случайного вектора
template <class T> xt::xarray<T> generate_vector(int n) {
  xt::xarray<T> b = xt::zeros<T>({n});
  for (int i = 0; i < n; ++i) {
    b(i) = dis(gen) * 10.0;
  }
  return b;
}

// Вычисление нормы невязки
template <class T>
T compute_residual_norm(const xt::xarray<T> &A, const xt::xarray<T> &b,
                        const xt::xarray<T> &x) {
  auto residual = b - xt::linalg::dot(A, x);
  return xt::linalg::norm(residual);
}

// Тестирование fixed_point_solver
template <class T> void test_fixed_point_solver(int n) {
  auto A = generate_fixed_point_matrix<T>(n);
  auto b = generate_vector<T>(n);
  xt::xarray<T> x0 = genbad(A, b);
  auto f = xt::zeros<T>({n}); // Добавляем параметр f

  auto start = std::chrono::high_resolution_clock::now();
  auto x = fixed_point_solver<T, 1000>(A, b, x0, 0.1, f);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();
  auto error = compute_residual_norm(A, b, x);

  std::cout << "fixed_point_solver " << n << " " << duration << " " << error
            << std::endl;
}

// Тестирование jacobi_solver
template <class T> void test_jacobi_solver(int n) {
  xt::xarray<T> A = generate_diagonally_dominant_matrix<T>(n);
  xt::xarray<T> b = generate_vector<T>(n);
  xt::xarray<T> x0 = genbad(A, b);

  auto start = std::chrono::high_resolution_clock::now();
  auto x = jacobi_solver(A, b, x0, 1e-6);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();
  auto error = compute_residual_norm(A, b, x);

  std::cout << "jacobi_solver " << n << " " << duration << " " << error
            << std::endl;
}

// Тестирование seidel_solver
template <class T> void test_seidel_solver(int n) {
  xt::xarray<T> A = generate_diagonally_dominant_matrix<T>(n);
  xt::xarray<T> b = generate_vector<T>(n);
  xt::xarray<T> x0 = genbad(A, b);

  auto start = std::chrono::high_resolution_clock::now();
  auto x = seidel_solver(A, b, x0, 1e-6, 1000);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();
  auto error = compute_residual_norm(A, b, x);

  std::cout << "seidel_solver " << n << " " << duration << " " << error
            << std::endl;
}

// Тестирование minres_solver
template <class T> void test_minres_solver(int n) {
  xt::xarray<T> A = generate_spd_matrix<T>(n);
  xt::xarray<T> b = generate_vector<T>(n);
  xt::xarray<T> x0 = genbad(A, b);

  auto start = std::chrono::high_resolution_clock::now();
  auto x = minres_solver(A, b, x0, 1e-6, 1000);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();
  auto error = compute_residual_norm(A, b, x);

  std::cout << "minres_solver " << n << " " << duration << " " << error
            << std::endl;
}

// Основная функция тестирования
void run_benchmarks(int dimension) {

  test_fixed_point_solver<double>(dimension);
  test_jacobi_solver<double>(dimension);
  test_seidel_solver<double>(dimension);
  test_minres_solver<double>(dimension);
}

#endif // !__TOOLS_HPP__
