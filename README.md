# Car Horsepower Multilinear Regression using PySpark

PySpark for multiple linear regression on car horsepower using SMOTE for data augmentation.

---

## Problem Overview

Multilinear regression is a linear regression using more than 1 independent variable to predict the dependent variable. In the context where multiple variables can be used to train a regressor, [feature selection] makes it possible to respect the [data characteristics assumed by linear regressors]: linear relationship between predictor and predicted variables, multivariate normality, homoscedasticity, no auto-correlation and no collinearity.

Our objective is to build a regressor that is able to predict a car's horsepower given its characteristics. In the [dataset] used for this analysis, multiple features are available. Some of them have little to no correlation with the predicted variable "HP". Some features also have high correlation with each other, indicating the existence of collinearity. Nonetheless, the most harsh limitation of this dataset is its very small size. The 32 instances of our dataset have 1 predicted ("HP") and 10 predictor features related to common characteristics of a car:

| Attribute | Summary |
|:---------:|:-------:|
| MilesPerGallon | Number of miles the car can travel per US gallon. |
| Cylinders | Number of cylinders in engine. |
| Displacement | Cylinder volume swept by all of the pistons in cubic inches. |
| RearAxleRatio | Rear axle ratio. |
| Weight | Car weight (lb / 1000). |
| QuarterMileTime | Time the car takes to make a quarter of a mile in seconds. |
| VShapeOrStraightLine | If the cylinder configuration follows a V shape or a straight line. |
| AutomaticOrManual | If transmission is automatic (= 0) or manual (= 1). |
| Gears | Number of forward gears. |
| Carburetor | Number of carburetors. |
| HP | Gross car horsepower. |

## Analysis Introduction

Our approach for this analysis is to first create 2 regressors studying only the correlation between predictor and predicted variables. Later, we instantiate another model also taking the collinearity aspect into consideration. It becomes very clear that removing variables is beneficial if the removed variables were presenting collinearity with other variables. 4 performance metrics are used to compare the 4 different regressors created during this analysis: [R2 score, Root Mean Squared Error (RMSE), Mean Absolute Error (MAE) and explained variance].

The intuition that any model trained on this dataset would be limited by its very small size is confirmed after experimenting with [data augmentation]. Using [SMOTE], a method capable of generating synthetic numerical data, we are able to increase our dataset up to 500 instances with a reasonable amount of balance between the 5 natural groupings formed by the "HP" predicted variable.

The data-driven data augmentation ability of SMOTE is paired with [PySpark]'s data-driven method for bucketizing data: [QuantileDiscretizer]. This method allows to separate the dataset into buckets using summary statistics, like median and quantiles. It is a necessary step since SMOTE was originally proposed to work with classification data. The buckets become our classes; used for data augmentation only.

Additionally, generalized linear regression is also used to train regressors assuming different error distributions of the data. Testing many different error distributions, the best regressor model was the one considering that errors followed a compound Poisson-Gamma distribution.

The final model is a regressor assuming a Poisson-Gamma error distribution, using only variables with no (or small) collinearity and trained on the augmented dataset. **This model was able to achieve a very high R2 score of around 0.93**, while maintaining a low MAE of roughly 12. This error rate is less than 5% of the predicted variable range (283).

[//]: #

[dataset]: <https://www.rdocumentation.org/packages/datasets/versions/3.6.2/topics/mtcars>
[feature selection]: <https://en.wikipedia.org/wiki/Feature_selection>
[data characteristics assumed by linear regressors]: <https://www.statisticssolutions.com/free-resources/directory-of-statistical-analyses/assumptions-of-linear-regression/>
[PySpark]: <https://spark.apache.org/docs/latest/api/python/>
[data augmentation]: <https://www.tensorflow.org/tutorials/images/data_augmentation>
[QuantileDiscretizer]: <https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.QuantileDiscretizer.html>
[SMOTE]: <https://arxiv.org/pdf/1106.1813.pdf>
[R2 score, Root Mean Squared Error (RMSE), Mean Absolute Error (MAE) and explained variance]: <https://spark.apache.org/docs/latest/mllib-evaluation-metrics.html#regression-model-evaluation>
