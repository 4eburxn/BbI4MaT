#include "lib/itersolver.hpp"
#include "lib/norm.hpp"
#include "lib/solver.hpp"
#include "lib/tools.hpp"
#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/core/xtensor_forward.hpp"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/core/xoperation.hpp>
#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/generators/xrandom.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/views/xslice.hpp>
#include <xtensor/views/xview.hpp>
int main(int argc, char *argv[]) {
  for (int i = 5000; i < 10000; i += 100) {
    xt::xarray<double> rnd = xt::random::randn<double>({i * 3 + 1});
    xt::xarray<double> mtr = xt::zeros<double>({i, i + 1});
    mtr(0, 0) = rnd(0);
    mtr(0, 1) = rnd(1);
    mtr(i - 1, i - 2) = rnd(i * 3);
    mtr(i - 1, i - 1) = rnd(i * 3 - 1);
    xt::view(mtr, xt::all(), i) = xt::random::randn<double>({i});
    for (int j = 1; j < i - 1; j++) {
      mtr(j, j) = rnd(j * 3);
      mtr(j, j - 1) = rnd(j * 3 + 1);
      mtr(j, j + 1) = rnd(j * 3 - 1);
    }
    auto start = std::chrono::high_resolution_clock::now();
    auto p = tridiagonal_solver(mtr);
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = finish - start;
    std::cout << "TriD " << i << " " << elapsed.count() << std::endl;
  }

  // mtr(0, 3) = mtr(1, 3) = mtr(2, 3) = 1;
  // std::cout << Residual(xt::xarray<double>(
  //                           xt::view(mtr, xt::all(), xt::range(0, 3))),
  //                       oth_solver(mtr),
  //                       xt::xarray<double>(xt::view(mtr, xt::all(), 3)))
  //           << "\n"
  //           << unordered_gauss_solver(mtr) << std::endl;

  return 0;

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
