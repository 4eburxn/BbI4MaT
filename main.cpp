// #define XTENSOR_USE_XSIMD 1
// #define XTENSOR_USE_OPENMP 1
#include "lib/norm.hpp"
#include "lib/solver.hpp"
#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/core/xmath.hpp"
#include "xtensor/core/xtensor_forward.hpp"
#include "xtensor/misc/xmanipulation.hpp"
#include "xtensor/views/xstrided_view.hpp"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/generators/xrandom.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/views/xview.hpp>
int main(int argc, char *argv[]) {
  using namespace xt::placeholders;
  xt::xarray<double> arr1{
      {1.0, -3.0, 2.0, 1.}, {4.0, 5.0, -1.0, 2.}, {3.0, 8.0, -6.0, 3.}};
  xt::random::seed(227);
  xt::xarray<double> m = xt::random::randn<double>({2000, 2000});
  xt::xarray<double> b = xt::random::randn<double>({2000});
  // xt::xarray<double> arr1{
  //     {1.0, 0, 0, 1.}, {4.0, 5.0, 0, 2.}, {3.0, 8.0, -6.0, 3.}};

  xt::xarray<double> arr2{1, 2, 3};
  xt::xarray<double> arr3{4, 5, 6};

  // xt::xarray<double> res = xt::view(arr1, 1) + arr2;

  std::cout << xt::linalg::vdot(xt::view(arr2, xt::all()), arr3) << std::endl;
  std::cout << arr1 << std::endl;
  std::cout << solver(arr1, arr2) << std::endl;
  auto LU = lu_decomposition(arr1);
  std::cout << LU.first << "\n" << LU.second << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
  // auto p = lu_decomposition(m);
  auto p = solver(m, b);
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = finish - start;
  std::cout << "Elapsed Time: " << elapsed.count() << " milseconds"
            << std::endl;
  return 0;
}
