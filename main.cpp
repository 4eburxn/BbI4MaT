#include "lib/norm.hpp"
#include "lib/solver.hpp"
#include "xtensor/core/xmath.hpp"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/generators/xrandom.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/views/xview.hpp>
int main(int argc, char *argv[]) {
  // xt::xarray<double> arr1{
  //     {1.0, -3.0, 2.0, 1.}, {4.0, 5.0, -1.0, 2.}, {3.0, 8.0, -6.0, 3.}};
  xt::random::seed(227);
  xt::xarray<double> m = xt::random::randn<double>({10000, 10001});
  xt::xarray<double> arr1{
      {1.0, 0, 0, 1.}, {4.0, 5.0, 0, 2.}, {3.0, 8.0, -6.0, 3.}};

  xt::xarray<double> arr2{-5.0, 6.0, 7};

  // xt::xarray<double> res = xt::view(arr1, 1) + arr2;

  std::cout << arr1 << std::endl;
  std::cout << unordered_gauss_solver(arr1) << std::endl;
  // std::cout << gauss_solver(m) << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
  auto p = gauss_solver(m);
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = finish - start;
  std::cout << "Elapsed Time: " << elapsed.count() << " milseconds"
            << std::endl;
  return 0;
}
