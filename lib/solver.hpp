#ifndef __SOLVER_HPP__
#define __SOLVER_HPP__
#include "xtensor/core/xmath.hpp"
#include "xtensor/core/xtensor_forward.hpp"
#include "xtensor/misc/xmanipulation.hpp"
#include "xtensor/views/xslice.hpp"
#include <cmath>
#include <iostream>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/views/xview.hpp>

enum SolverType { UnorderedGauss, Gauss, Orthogonalization, tridiagonal };

template <class T>
inline xt::xarray<T> unordered_gauss_solver(xt::xarray<T> tensor) {
  using namespace xt::placeholders;
  int n = tensor.shape()[0];

  for (int i = 0; i < n - 1; i++) {
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
  return tensor;
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
  return tensor;
}

#endif // __SOLVER_HPP__
