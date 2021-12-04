//
// Created by f3ar13ss on 04.12.2021.
//

#ifndef MACHINELEARNINGCPP_LINEARREGRESSION_H
#define MACHINELEARNINGCPP_LINEARREGRESSION_H

#include "submodules/eigen/Eigen/Dense"
#include <vector>

using namespace std;

class LinearRegression {

public:
    LinearRegression()
    {}

    float Cost(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd theta);
    std::tuple<Eigen::VectorXd, vector<float>> GradientDescent(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::VectorXd theta, float alpha, int iters);
    float RSquared(Eigen::MatrixXd y, Eigen::MatrixXd y_hat);
};


#endif //MACHINELEARNINGCPP_LINEARREGRESSION_H
