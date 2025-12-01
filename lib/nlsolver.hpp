#ifndef __NLSOLVER_HPP__
#define __NLSOLVER_HPP__

#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xarray.hpp>

double bisection(std::function<double(double)> f, double a, double b,
                 double eps = 1e-6, int max_iter = 100) {
  for (int i = 0; i < max_iter; ++i) {
    double c = (a + b) / 2.0;
    if (std::abs(b - a) < eps || std::abs(f(c)) < 1e-12) {
      return c;
    }
    if (f(a) * f(c) < 0) {
      b = c;
    } else {
      a = c;
    }
  }
  return (a + b) / 2.0;
}

double newton(std::function<double(double)> f, double x0, double eps = 1e-6,
              int max_iter = 100) {
  double h = 1e-5;
  double x = x0;

  for (int i = 0; i < max_iter; ++i) {
    double fx = f(x);
    double df = (f(x + h) - f(x - h)) / (2.0 * h);

    if (std::abs(df) < 1e-12) {
      break;
    }

    double x_new = x - fx / df;

    if (std::abs(x_new - x) < eps) {
      return x_new;
    }
    x = x_new;
  }
  return x;
}

xt::xarray<double> newton_multidimensional(
    const std::vector<std::function<double(const xt::xarray<double> &)>> &F,
    const xt::xarray<double> &x0, double eps = 1e-6, int max_iter = 100) {

  int n = x0.size();
  xt::xarray<double> x = x0;
  double h = 1e-5;

  for (int iter = 0; iter < max_iter; ++iter) {
    xt::xarray<double> Fx = xt::zeros<double>({n});
    for (int i = 0; i < n; ++i) {
      Fx[i] = F[i](x);
    }

    xt::xarray<double> J = xt::zeros<double>({n, n});

    for (int j = 0; j < n; ++j) {
      xt::xarray<double> dx = xt::zeros<double>(x.shape());
      dx[j] = h;

      xt::xarray<double> F_forward = xt::zeros<double>({n});
      xt::xarray<double> F_backward = xt::zeros<double>({n});

      for (int i = 0; i < n; ++i) {
        F_forward[i] = F[i](x + dx);
        F_backward[i] = F[i](x - dx);
      }

      xt::view(J, xt::all(), j) = (F_forward - F_backward) / (2.0 * h);
    }

    double cond = xt::linalg::cond(J, 2.);
    if (cond > 1e12) {
      std::cout << "Warning: Ill-conditioned Jacobian (cond = " << cond << ")"
                << std::endl;
      break;
    }

    xt::xarray<double> delta = xt::linalg::solve(J, -Fx);
    xt::xarray<double> x_new = x + delta;

    double norm = xt::linalg::norm(delta, 2);
    if (norm < eps) {
      return x_new;
    }

    x = x_new;
  }

  return x;
}

double simple_iteration(std::function<double(double)> phi, double x0,
                        double eps = 1e-6, int max_iter = 100) {
  double x = x0;
  double h = 1e-5;

  for (int i = 0; i < max_iter; ++i) {
    double derivative = (phi(x + h) - phi(x - h)) / (2.0 * h);
    if (std::abs(derivative) > 1.0) {
      std::cout << "simple_iteration: does not converge" << std::endl;
      break;
    }

    double x_new = phi(x);
    if (std::abs(x_new - x) < eps) {
      return x_new;
    }
    x = x_new;
  }

  return x;
}

// вспомогательная функция для создания многомерных функций
template <typename... Args>
auto make_function(std::function<double(Args...)> f) {
  return [f](const xt::xarray<double> &x) -> double { return f(x[0]); };
}

#endif // !__NLSOLVER_HPP__
