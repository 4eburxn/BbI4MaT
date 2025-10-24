#ifndef __SOLVER_HPP__
#define __SOLVER_HPP__
#include "xtensor/core/xtensor_forward.hpp"
#include "xtensor/generators/xbuilder.hpp"
#include "xtensor/misc/xmanipulation.hpp"
#include "xtensor/views/xslice.hpp"
#include <cmath>
#include <omp.h>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/views/xview.hpp>

enum SolverType {
  Default = 0,
  UnorderedGauss = 1,
  Gauss = 2,
  Orthogonalization = 3,
  tridiagonal = 4
};

template <class T>
inline xt::xarray<T> unordered_gauss_solver(xt::xarray<T> tensor) {
  using namespace xt::placeholders;
  int n = tensor.shape()[0];

  for (int i = 0; i != n - 1; i++) {
    auto view_ = xt::range(i, _);
    auto v = xt::view(tensor, i, view_);
    for (int j = i + 1; j < n; j++) {
      auto vv = xt::view(tensor, j, view_);
      vv -= (tensor(j, i) / tensor(i, i)) * v;
    }
  }
  for (int i = n - 1; i != 0; i--) {
    auto view_ = xt::range(i, _);
    auto v = xt::view(tensor, i, view_);
    v = v / tensor(i, i);
    for (int j = i - 1; j >= 0; j--) {
      auto vv = xt::view(tensor, j, view_);
      vv -= (tensor(j, i)) * v;
    }
  }
  return xt::view(tensor, xt::all(), n);
}

template <class T> inline xt::xarray<T> gauss_solver(xt::xarray<T> tensor) {
  using namespace xt::placeholders;
  int n = tensor.shape()[0];
  xt::xarray<T> tmp(n + 1);

  for (int i = 0; i < n - 1; i++) {
    auto mxi = i;
    auto mx = std::abs(tensor(i, i));
    for (int j = i + 1; j < n; j++) {
      if (std::abs(tensor(j, i)) > mx) {
        mxi = j;
        mx = std::abs(tensor(j, i));
      }
    }
    tmp = xt::view(tensor, i);
    xt::view(tensor, i) = xt::view(tensor, mxi);
    xt::view(tensor, mxi) = tmp;
    auto view_ = xt::range(i, _);
    auto v = xt::view(tensor, i, view_);
    for (int j = i + 1; j < n; j++) {
      auto vv = xt::view(tensor, j, view_);
      vv -= (tensor(j, i) / tensor(i, i)) * v;
    }
  }
  for (int i = n - 1; i != 0; i--) {
    auto view_ = xt::range(i, _);
    auto v = xt::view(tensor, i, view_);
    v = v / tensor(i, i);
    for (int j = i - 1; j >= 0; j--) {
      auto vv = xt::view(tensor, j, view_);
      vv -= (tensor(j, i)) * v;
    }
  }
  return xt::view(tensor, xt::all(), n);
}

template <class T>
inline std::pair<xt::xarray<T>, xt::xarray<T>>
lu_decomposition(const xt::xarray<T> &A) {
  using namespace xt::placeholders;
  size_t n = A.shape()[0];
  xt::xarray<T> L = xt::eye<T>({n, n});
  xt::xarray<T> U = xt::zeros<T>({n, n});

  for (size_t i = 0; i < n; ++i) {
    auto ppp = xt::range(_, i);
    auto L_i = xt::view(L, i, ppp);
    for (size_t j = i; j < n; ++j) {
      auto U_col_j = xt::view(U, ppp, j);
      U(i, j) = A(i, j) - xt::linalg::vdot(L_i, U_col_j);
    }

    T U_diag = U(i, i);
    auto vvv = xt::range(i + 1, _);
    auto L_col_i = xt::view(L, vvv, i);
    auto A_col_i = xt::view(A, vvv, i);

    for (size_t k = i + 1; k < n; ++k) {
      auto L_k = xt::view(L, k, ppp);
      auto U_col_i = xt::view(U, ppp, i);
      L(k, i) = (A(k, i) - xt::linalg::vdot(L_k, U_col_i)) / U_diag;
    }
  }
  return {L, U};
}

template <class T>
inline xt::xarray<T> solver_lu_based(xt::xarray<T> tensor, xt::xarray<T> b) {
  using namespace xt;
  using namespace xt::placeholders;
  size_t n = tensor.shape()[0];
  xt::xarray<T> L = xt::eye<T>({n, n});
  xt::xarray<T> U = xt::zeros<T>({n, n});

  for (size_t i = 0; i < n; ++i) {
    auto ppp = xt::range(_, i);
    auto L_i = xt::view(L, i, ppp);
    for (size_t j = i; j < n; ++j) {
      auto U_col_j = xt::view(U, ppp, j);
      U(i, j) = tensor(i, j) - xt::linalg::vdot(L_i, U_col_j);
    }

    T U_diag = U(i, i);
    auto vvv = xt::range(i + 1, _);
    auto L_col_i = xt::view(L, vvv, i);
    auto A_col_i = xt::view(tensor, vvv, i);

    for (size_t k = i + 1; k < n; ++k) {
      auto L_k = xt::view(L, k, ppp);
      auto U_col_i = xt::view(U, ppp, i);
      L(k, i) = (tensor(k, i) - xt::linalg::vdot(L_k, U_col_i)) / U_diag;
    }
  }

  for (int i = 0; i != n - 1; i++) {
    b(i) /= L(i, i);
    for (int j = i + 1; j < n; j++) {
      b(j) -= b(i) * (L(j, i));
    }
  }
  for (int i = n - 1; i != 0; i--) {
    b(i) /= U(i, i);
    for (int j = i - 1; j >= 0; j--) {
      b(j) -= b(i) * (U(j, i));
    }
  }
  return b;
}

template <class T>
inline xt::xarray<T> solver(xt::xarray<T> tensor, xt::xarray<T> b,
                            SolverType Type = SolverType::Default) {
  xt::xarray<T> A = xt::zeros<T>({tensor.shape()[0], tensor.shape()[0] + 1});
  switch (Type) {
  case SolverType::Default:
    return solver_lu_based(tensor, b);
  case SolverType::UnorderedGauss:
    xt::view(A, xt::all(), xt::range(xt::placeholders::_, tensor.shape()[0])) =
        tensor;
    xt::view(A, xt::all(), tensor.shape()[0]) = b;
    return unordered_gauss_solver(A);
  case SolverType::Gauss:
    xt::view(A, xt::all(), xt::range(xt::placeholders::_, tensor.shape()[0])) =
        tensor;
    xt::view(A, xt::all(), tensor.shape()[0]) = b;
    return gauss_solver(A);
  }
}

#endif // __SOLVER_HPP__
