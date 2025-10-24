#ifndef __TOOLS_HPP__
#define __TOOLS_HPP__
#include "xtensor/core/xtensor_forward.hpp"
#include "xtensor/views/xslice.hpp"
#include <cmath>
#include <omp.h>
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

#endif // !__TOOLS_HPP__
