#include "lib/itersolver.hpp"
#include "lib/nlsolver.hpp"
#include "lib/norm.hpp"
#include "lib/solver.hpp"
#include "lib/tools.hpp"
#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/core/xtensor_forward.hpp"
#include "xtensor/misc/xcomplex.hpp"
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/core/xoperation.hpp>
#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/generators/xrandom.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/views/xslice.hpp>
#include <xtensor/views/xview.hpp>

int main(int argc, char *argv[]) {

  auto f1 = [](double x) { return x * x - 2.0; };
  double root1 = bisection(f1, 0.0, 2.0);
  std::cout << "Bisection x * x - 2.0;  x = " << root1 << std::endl;

  double root2 = newton(f1, 1.0);
  std::cout << "Newton x * x - 2.0;  x = " << root2 << std::endl;

  auto phi = [](double x) { return std::cos(x); };
  double root3 = simple_iteration(phi, 0.5);
  std::cout << "Simple iteration cos(x)=x;  x = " << root3 << std::endl;

  // f1(x,y) = x^2 + y^2 - 4
  // f2(x,y) = exp(x) + y - 1
  std::vector<std::function<double(const xt::xarray<double> &)>> F2 = {
      [](const xt::xarray<double> &v) {
        return v[0] * v[0] + v[1] * v[1] - 4.0;
      },
      [](const xt::xarray<double> &v) { return std::exp(v[0]) + v[1] - 1.0; }};

  xt::xarray<double> x02 = {1.0, 1.0};
  xt::xarray<double> root4 = newton_multidimensional(F2, x02);
  std::cout << "Multidimensional Newton root: " << root4 << std::endl;

  return 0;

#if 0
  std::cout << "\n1. Решение уравнения x^2 - 4 = 0:" << std::endl;
  double root1 = newton_method_autodiff(
      quadratic_function<decltype(bm::make_fvar<double, 1>(0))>, 3.0);
  std::cout << "Найденный корень: " << root1 << std::endl;

  std::cout << "\n2. Решение уравнения x^3 - 2x - 5 = 0:" << std::endl;
  double root2 = newton_method_autodiff(
      cubic_function<decltype(bm::make_fvar<double, 1>(0))>, 2.0);
  std::cout << "Найденный корень: " << root2 << std::endl;

  std::cout << "\n3. Решение уравнения cos(x) - x = 0:" << std::endl;
  double root3 = newton_method_autodiff(
      cos_function<decltype(bm::make_fvar<double, 1>(0))>, 1.0);
  std::cout << "Найденный корень: " << root3 << std::endl;

  return 0;

  for (int dim = 3; dim <= 1000; dim += 30) {
    run_benchmarks(dim);
  }
  for (int dim = 1000; dim <= 10000; dim += 500) {
    run_benchmarks(dim);
  }
  return 0;
#endif

#if 0
  auto a = read_from_stdin<double>();
  auto N = a.second.shape()[0];
  xt::xarray<double> A = xt::zeros<double>({N, N + 1});
  xt::view(A, xt::all(), xt::range(xt::placeholders::_, N)) = a.first;
  xt::view(A, xt::all(), N) = a.second;

  auto start = std::chrono::high_resolution_clock::now();
  auto p = unordered_gauss_solver(A);
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = finish - start;
  std::cout << "Ugaus " << a.second.shape()[0] << " " << elapsed.count() << " "
            << normV<1, double>(Residual(a.first, p, a.second)) << std::endl;
  start = std::chrono::high_resolution_clock::now();
  p = gauss_solver(A);
  finish = std::chrono::high_resolution_clock::now();
  elapsed = finish - start;
  std::cout << "Gaus " << a.second.shape()[0] << " " << elapsed.count() << " "
            << normV<1, double>(Residual(a.first, p, a.second)) << std::endl;
  start = std::chrono::high_resolution_clock::now();
  p = solver_lu_based(a.first, a.second);
  finish = std::chrono::high_resolution_clock::now();
  elapsed = finish - start;
  std::cout << "LU " << a.second.shape()[0] << " " << elapsed.count() << " "
            << normV<1, double>(Residual(a.first, p, a.second)) << std::endl;
  start = std::chrono::high_resolution_clock::now();
  p = xt::linalg::solve(a.first, a.second);
  finish = std::chrono::high_resolution_clock::now();
  elapsed = finish - start;
  std::cout << "XT::BLAS " << a.second.shape()[0] << " " << elapsed.count()
            << " " << normV<1, double>(Residual(a.first, p, a.second))
            << std::endl;
  if (a.second.shape()[0] < 4600) {
    start = std::chrono::high_resolution_clock::now();
    p = oth_solver(A);
    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;
  } else {
    elapsed = start - start;
  }
  std::cout << "Oth " << a.second.shape()[0] << " " << elapsed.count() << " "
            << normV<1, double>(Residual(a.first, p, a.second)) << std::endl;

  auto i = a.second.shape()[0];
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
  start = std::chrono::high_resolution_clock::now();
  p = tridiagonal_solver(mtr);
  finish = std::chrono::high_resolution_clock::now();
  elapsed = finish - start;
  std::cout << "TriD " << i << " " << elapsed.count() << " "
            << normV<1, double>(
                   Residual(xt::xarray<double>(
                                xt::view(mtr, xt::all(), xt::range(0, i))),
                            p, xt::xarray<double>(xt::view(mtr, xt::all(), i))))
            << std::endl;
#endif

  return 0;
}
