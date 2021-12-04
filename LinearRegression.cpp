//
// Created by f3ar13ss on 04.12.2021.
//

#include "LinearRegression.h"

using namespace std;

// Cost function to analyze model
float LinearRegression::Cost(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd theta) {

    Eigen::MatrixXd inner = pow(((X*theta)-y).array(), 2);

    return (inner.sum() / (2*X.rows()));
}

std::tuple<Eigen::VectorXd, std::vector<float>> LinearRegression::GradientDescent(Eigen::MatrixXd X, Eigen::MatrixXd y,
                                                                                  Eigen::VectorXd theta, float alpha,
                                                                                  int iters) {
    Eigen::MatrixXd temp = theta;

    int parameters = theta.rows();

    std::vector<float> cost;
    cost.push_back(Cost(X, y, theta));

    for(int i=0; i<iters; ++i) {
        Eigen::MatrixXd error = X*theta - y;
        for(int j = 0; j < parameters; ++j) {
            Eigen::MatrixXd X_i = X.col(j);
            Eigen::MatrixXd term = error.cwiseProduct(X_i);
            temp(j, 0) = theta(j, 0) - ((alpha / X.rows()) * term.sum());
        }
        theta = temp;
        cost.push_back(Cost(X, y, theta));
    }

    return make_tuple(theta, cost);
}

float LinearRegression::RSquared(Eigen::MatrixXd y, Eigen::MatrixXd y_hat) {
    auto num = pow((y-y_hat).array(), 2).sum();
    auto den = pow(y.array()-y.mean(), 2).sum();

    return 1 - num/den;
}