#! usr/bin/env python3


from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

x_list = [1, 2, 3, 4, 5, 6]
y_list = [5, 4, 6, 5, 6, 7]
xs = np.array(x_list, dtype=np.float64)
ys = np.array(y_list, dtype=np.float64)


def best_fit(xs, yx):
    """Return best fit slope and y-intercept of xs and ys."""
    xs_bar = mean(xs)
    ys_bar = mean(ys)
    x_y_bar = mean(xs * ys)
    x_sq_bar = mean(xs ** 2)

    slope = (  (xs_bar * ys_bar - x_y_bar) /
             (xs_bar ** 2 - x_sq_bar)  )
    y_int = ys_bar - slope * xs_bar

    return slope, y_int


def squared_error(ys_orig, ys_line):
    """Return Squared Error of LBF."""
    squared_error = sum((ys_line - ys_orig) ** 2)
    return squared_error


def coefficient_of_determination(ys_orig, ys_line):
    """Return coefficient of determination (R^2)"""
    ys_mean = mean(ys_orig)
    y_mean_line = [ys_mean for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    r_squared = 1 - (squared_error_regr / squared_error_y_mean)
    return r_squared


# Find line of best fit
bf_line = best_fit(xs, ys)
slope = bf_line[0]
y_int = bf_line[1]


# Make regression line
regression_line = [(slope * x) + y_int for x in xs]


# Make predictions
predict_x = 8
predict_y = slope * predict_x + y_int


# Check accuracy
r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)


# Plot regression line and data
plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color='r')
plt.plot(xs, regression_line)


plt.show()
