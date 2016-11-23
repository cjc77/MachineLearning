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


ans = best_fit(xs, ys)
slope = ans[0]
y_int = ans[1]


# Make regression line
regression_line = [(slope * x) + y_int for x in xs]


# Make predictions
predict_x = 8
predict_y = slope * predict_x + y_int


# Plot regression line and data
plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color='r')
plt.plot(xs, regression_line)


plt.show()
