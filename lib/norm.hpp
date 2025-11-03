#ifndef __NORM_HPP__
#define __NORM_HPP__
#include <cmath>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/views/xview.hpp>

template <double N, class T> auto normV(xt::xarray<T> tensor) {
  using namespace xt;
  static_assert(N > 0 || N == -1., "normM: norm must be > 0 OR -1");

  if constexpr (N == 1.)
    return sum(abs(tensor));
  if constexpr (N == -1.)
    return amax(tensor);
  return pow(sum(pow(abs(tensor), N)), 1 / N);
}
template <int N, class T> auto normV(xt::xarray<T> tensor) {
  using namespace xt;
  static_assert(N < 0 || N != -1, "normM: norm must be > 0 OR -1");

  if constexpr (N == 1)
    return xt::sum(xt::abs(tensor));
  if constexpr (N == -1)
    return amax(tensor);
  // return pow(sum(pow(abs(tensor), (double)N)), 1 / (double)N);
}

template <int N, class T> auto normM(xt::xarray<T> tensor) {
  using namespace xt;
  static_assert(-2 < N && N < 3,
                "normM: norms, others than -1,1,2 are not implemented");

  if constexpr (N == 1)
    return amax(sum(abs(tensor), {1}));
  if constexpr (N == -1)
    return amax(sum(abs(tensor), {0}));
  return amax(diag(std::get<1>(xt::linalg::svd(tensor))));
}

#endif // __NORM_HPP__
