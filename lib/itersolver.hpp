
#ifndef __ITERSOLVER_HPP__
#define __ITERSOLVER_HPP__
#include "xtensor/core/xtensor_forward.hpp"
#include "xtensor/generators/xbuilder.hpp"
#include "xtensor/views/xslice.hpp"
#include <cmath>
#include <iostream>
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
  std::cout << B << "\n" << C << std::endl;

  int n = N;
  while (n--) {
    f0 = xt::linalg::dot(B, f0) + C;
  }
  return -f0;
}

#endif // __ITERSOLVER_HPP__
