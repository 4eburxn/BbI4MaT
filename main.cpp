#include "lib/norm.hpp"
#include "lib/solver.hpp"
#include "xtensor/core/xmath.hpp"
#include <iostream>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/views/xview.hpp>
int main(int argc, char *argv[]) {
  // xt::xarray<double> arr1{
  //     {1.0, -3.0, 2.0, 1.}, {4.0, 5.0, -1.0, 2.}, {3.0, 8.0, -6.0, 3.}};
  xt::xarray<double> arr1{
      {1.0, 0, 0, 1.}, {4.0, 5.0, 0, 2.}, {3.0, 8.0, -6.0, 3.}};

  xt::xarray<double> arr2{-5.0, 6.0, 7};

  // xt::xarray<double> res = xt::view(arr1, 1) + arr2;

  std::cout << arr1 << std::endl;
  std::cout << unordered_gauss_solver(arr1) << std::endl;
  std::cout << gauss_solver(arr1) << std::endl;

  return 0;
}
