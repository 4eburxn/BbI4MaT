#ifndef __NLSOLVER_HPP__
#define __NLSOLVER_HPP__

#include <boost/math/differentiation/autodiff.hpp>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#undef DEBUG
namespace bm = boost::math::differentiation;
template <typename Function>
double newton_method_autodiff(Function f, double x0, double tolerance = 1e-10,
                              int max_iterations = 100) {
  double x = x0;

#ifdef DEBUG
  std::cout << "Начальное приближение: x0 = " << x0 << std::endl;
  std::cout << std::setw(3) << "k" << std::setw(15) << "x_k" << std::setw(15)
            << "f(x_k)" << std::setw(15) << "f'(x_k)" << std::setw(15)
            << "|x_k - x_{k-1}|" << std::endl;
  std::cout << std::string(63, '-') << std::endl;
#endif // DEBUG

  for (int k = 0; k < max_iterations; ++k) {
    auto x_auto = bm::make_fvar<double, 1>(x);

    auto result = f(x_auto);
    double fx = result.derivative(0);
    double fpx = result.derivative(1);

    if (std::abs(fpx) < 1e-15) {
      break;
    }

    double x_new = x - fx / fpx;
    double dx = std::abs(x_new - x);

#ifdef DEBUG
    std::cout << std::setw(3) << k << std::setw(15) << x << std::setw(15) << fx
              << std::setw(15) << fpx << std::setw(15) << dx << std::endl;
#endif
    if (dx < tolerance && std::abs(fx) < tolerance) {
      return x_new;
    }

    x = x_new;
  }

#ifdef DEBUG
  std::cout << "Достигнуто максимальное количество итераций" << std::endl;
#endif
  return x;
}

#endif // !__NLSOLVER_HPP__
