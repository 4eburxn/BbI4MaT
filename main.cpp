#include "lib/norm.hpp"
#include "xtensor/core/xmath.hpp"
#include <iostream>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/views/xview.hpp>
int main(int argc, char *argv[]) {
  xt::xarray<double> arr1{{1.0, -3.0, 2.0}, {4.0, 5.0, -1.0}, {3.0, 8.0, -6.0}};

  xt::xarray<double> arr2{-5.0, 6.0, 7};

  xt::xarray<double> res = xt::view(arr1, 1) + arr2;

  std::cout << normM<2>(arr1) << std::endl;

  return 0;
}
