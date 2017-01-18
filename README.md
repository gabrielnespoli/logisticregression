# logisticregression
logistic regression implementation which learns from a training set and makes the classification of the test set

This algorithm can be used with any number of columns.

It's expected that the columns are normalized, though, a normalize function is available.

Two dataset are available to test the algorithm: training.csv and test.csv. They were extracted from the framingham.csv. The test set is the last 20 lines without the Y column (resultant column - prediction). The training lines are all framingham lines but the last 20 lines.
