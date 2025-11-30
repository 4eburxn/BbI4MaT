#ifndef __ITERSOLVER_HPP__
#define __ITERSOLVER_HPP__
#include "xtensor/core/xtensor_forward.hpp"
#include "xtensor/generators/xbuilder.hpp"
#include <cmath>
#include <omp.h>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/views/xview.hpp>

template <class T, int N = 10000>
inline xt::xarray<T> fixed_point_solver(xt::xarray<T> &tensor, xt::xarray<T> &f,
                                        xt::xarray<T> &f0, T tao = 1,
                                        xt::xarray<T> P = xt::xarray<T>({1})) {
  if (P.shape()[0] == 1) {
    P = -xt::eye({tensor.shape()[0], tensor.shape()[0]}) * tao;
  }
  auto B = xt::linalg::dot(P, tensor) +
           xt::eye({tensor.shape()[0], tensor.shape()[0]});
  auto C = xt::linalg::dot(f, P);

  int n = N;
  while (n--) {
    f0 = xt::linalg::dot(B, f0) - C;
  }
  return f0;
}

// Реализация метода Якоби
template <class T, int N = 10000>
inline xt::xarray<T> jacobi_solver(xt::xarray<T> &A, xt::xarray<T> &b,
                                   xt::xarray<T> &x0, T tolerance = 1e-6) {
  int n = A.shape()[0];
  xt::xarray<T> x = x0;
  xt::xarray<T> x_new = xt::zeros_like(x0);

  for (int iter = 0; iter < N; ++iter) {
    for (int i = 0; i < n; ++i) {
      T sum = 0.0;
      for (int j = 0; j < n; ++j) {
        if (j != i) {
          sum += A(i, j) * x(j);
        }
      }
      x_new(i) = (b(i) - sum) / A(i, i);
    }

    T error = xt::linalg::norm(x_new - x);
    if (error < tolerance) {
      return x_new;
    }

    x = x_new;
  }

  return x_new;
}

template <class T>
inline xt::xarray<T> seidel_solver(xt::xarray<T> &A, xt::xarray<T> &b,
                                   xt::xarray<T> &x0, T tolerance = 1e-6,
                                   int max_iterations = 1000) {
  int n = A.shape()[0];
  xt::xarray<T> x = x0;
  xt::xarray<T> x_old = x0;

  for (int iter = 0; iter < max_iterations; ++iter) {
    x_old = x;

    for (int i = 0; i < n; ++i) {
      T sum = 0.0;

      for (int j = 0; j < i; ++j) {
        sum += A(i, j) * x(j);
      }

      for (int j = i + 1; j < n; ++j) {
        sum += A(i, j) * x_old(j);
      }

      x(i) = (b(i) - sum) / A(i, i);
    }

    T error = xt::linalg::norm(x - x_old);
    if (error < tolerance) {
      return x;
    }
  }

  return x;
}

template <class T>
inline xt::xarray<T> minres_solver(const xt::xarray<T> &A,
                                   const xt::xarray<T> &b,
                                   const xt::xarray<T> &x0, T tolerance = 1e-6,
                                   int max_iterations = 1000) {
  int n = A.shape()[0];
  xt::xarray<T> x = x0;

  xt::xarray<T> r = b - xt::linalg::dot(A, x);
  T norm_r = xt::linalg::norm(r);
  T norm_b = xt::linalg::norm(b);

  if (norm_b < tolerance) {
    norm_b = 1.0;
  }

  if (norm_r < tolerance * norm_b) {
    return x;
  }

  xt::xarray<T> v_old = xt::zeros_like(b);
  xt::xarray<T> v = r / norm_r;
  xt::xarray<T> w = xt::zeros_like(b);
  xt::xarray<T> w_old = xt::zeros_like(b);

  T beta = norm_r;
  T beta_old = 0.0;
  T beta_new = 0.0;

  T c = 1.0;
  T c_old = 1.0;
  T s = 0.0;
  T s_old = 0.0;

  T eta = norm_r;

  xt::xarray<T> Av;

  for (int iter = 0; iter < max_iterations; ++iter) {
    Av = xt::linalg::dot(A, v);

    T alpha = xt::linalg::dot(v, Av)(0);

    if (iter == 0) {
      Av = Av - alpha * v;
    } else {
      Av = Av - alpha * v - beta_old * v_old;
    }

    beta_new = xt::linalg::norm(Av);

    if (beta_new < tolerance) {
      xt::xarray<T> w_new = (v - (s_old * beta_old) * w_old) / c;
      x = x + (eta / c) * w_new;
      return x;
    }

    T rho0 = c_old * alpha - c * s_old * beta_old;
    T rho1 = std::sqrt(rho0 * rho0 + beta_new * beta_new);
    T rho2 = s_old * alpha + c * c_old * beta_old;
    T rho3 = s * beta_old;

    c_old = c;
    s_old = s;

    c = rho0 / rho1;
    s = beta_new / rho1;

    xt::xarray<T> w_new;
    if (iter == 0) {
      w_new = v / rho1;
    } else {
      w_new = (v - rho2 * w - rho3 * w_old) / rho1;
    }

    x = x + c * eta * w_new;

    eta = -s * eta;

    T relative_error = std::abs(eta) / norm_b;
    if (relative_error < tolerance) {
      return x;
    }

    v_old = v;
    v = Av / beta_new;

    w_old = w;
    w = w_new;

    beta_old = beta_new;
  }

  return x;
}

#endif // __ITERSOLVER_HPP__
