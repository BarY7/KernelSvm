from skeleton_svm import *


x_train, y_train, x_val, y_val = get_points()
rbf_accuracy_per_gamma(x_train, y_train, x_val, y_val)
