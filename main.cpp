// #define XTENSOR_USE_XSIMD 1
// #define XTENSOR_USE_OPENMP 1
#include "lib/itersolver.hpp"
#include "lib/norm.hpp"
#include "lib/solver.hpp"
#include "lib/tools.hpp"
#include "xtensor-blas/xlinalg.hpp"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <semaphore>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/generators/xrandom.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/views/xview.hpp>
int main(int argc, char *argv[]) {

  auto a = read_from_stdin<double>();
  auto N = a.second.shape()[0];
  xt::xarray<double> A = xt::zeros<double>({N, N + 1});
  xt::view(A, xt::all(), xt::range(xt::placeholders::_, N)) = a.first;
  xt::view(A, xt::all(), N) = a.second;

  auto start = std::chrono::high_resolution_clock::now();
  auto p = unordered_gauss_solver(A);
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = finish - start;
  std::cout << "Ugaus " << a.second.shape()[0] << " " << elapsed.count()
            << std::endl;
  start = std::chrono::high_resolution_clock::now();
  auto pp = gauss_solver(A);
  finish = std::chrono::high_resolution_clock::now();
  elapsed = finish - start;
  std::cout << "Gaus " << a.second.shape()[0] << " " << elapsed.count()
            << std::endl;
  start = std::chrono::high_resolution_clock::now();
  auto ppp = solver_lu_based(a.first, a.second);
  finish = std::chrono::high_resolution_clock::now();
  elapsed = finish - start;
  std::cout << "LU " << a.second.shape()[0] << " " << elapsed.count()
            << std::endl;
  start = std::chrono::high_resolution_clock::now();
  auto pppp = xt::linalg::solve(a.first, a.second);
  finish = std::chrono::high_resolution_clock::now();
  elapsed = finish - start;
  std::cout << "XT::BLAS " << a.second.shape()[0] << " " << elapsed.count()
            << std::endl;
  return 0;
}
