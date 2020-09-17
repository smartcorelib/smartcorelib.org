---
layout: manual
title:  "Supervised Learning"
---

# Supervised Learning

Most machine learning problems falls into one of two categories: _supervised_ and _unsupervised_.
This page describes supervised learning algorithms implemented in *SmartCore*. 

In supervised learning we build a model which can be written in the very general form as
\\[Y = f(X) + \epsilon\\]
\\(X = (X_1,X_2,...,X_p)\\) are observations that consist of \\(p\\) different predictors, \\(Y\\) is an associated target values and 
\\(\epsilon\\) is a random error term, which is independent of \\(X\\) and has zero mean.
We fit an unknown function \\(f\\) to our data to predict the response for future observations or better understand the relationship between the response and the predictors.

Supervised learning is by far the most common machine learning method that is used in practice. Supervised learning problems can be grouped into regression and classification:

* *Classification*: when the output variable is qualitative (category), such as "cancerous" or "benign", "round" or "square".
* *Regression*: a regression problem is when the output variable is quantitative, such as "height" or "dollars".

All algorithms in *SmartCore* support both, qualitative and quantitative response variables and follow the same naming convention for function names. 
To fit an algorithm to your data use `fit` method that usually comes with at least 2 mandatory parameters: `x` for your predictors and `y` for target values. All optional parameters are hidden behind `Default::default()`.
To make a prediction use `predict` method that takes new observations as `x` and predicts estimated class labels or target values. 

## K Nearest Neighbors 

K-nearest neighbors (KNN) is one of the simplest and best-known non-parametric classification and regression method.
KNN does not require training. The algorithm simply stores the entire dataset and then uses this dataset to make predictions.

More formally,
given a positive integer \\(K\\) and a test observation \\(x_0\\), the KNN classifier first identifies the \\(K\\) points in the training data that are closest to \\(x_0\\), represented by \\(N_0\\) and then estimates the conditional probability for class \\(j\\) as the fraction of points in N0 whose response values equal \\(j\\):

\\[ Pr(Y=j \vert X=x_0) = \frac{1}{K} \sum_{i \in N_0} I(y_i=j) \\]

KNN Regressor is closely related to the KNN classifier. It estimates target value using the average of all the reponses in \\(N_0\\), i.e.

\\[ \hat{y} = \frac{1}{K} \sum_{i \in N_0} y_i \\]

The choice of \\(K\\) is very important. \\(K\\) can be found by tuning algorithm on a holdout dataset. It is a good idea to try many different values for \\(K\\) (e.g. values from 1 to 21) and see which value gives the best test error rate.

To determine which of the \\(K\\) instances in the training dataset are most similar to a new input a [distance metric]({{site.api_base_url}}/math/distance/index.html) is used. 
For real-valued input variables, the most popular distance metric is [Euclidean distance]({{site.api_base_url}}/math/distance/euclidian/index.html). You can choose the best distance metric based on the properties of your data. If you are unsure, you can experiment with different distance metrics and different values of \\(K\\) together and see which mix results in the most accurate models.

### Nearest Neighbors Classification 

To fit KNN Classifier to your data use [`KNNClassifier`]({{site.api_base_url}}/neighbors/knn_classifier/struct.KNNClassifier.html). Let's fit KNN Classifier to the [Breast Cancer]({{site.api_base_url}}/dataset/breast_cancer/index.html) dataset:

```rust
use smartcore::dataset::*;
// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
// Imports for KNN classifier
use smartcore::math::distance::Distances;
use smartcore::neighbors::knn_classifier::*;
// Model performance
use smartcore::metrics::roc_auc_score;
use smartcore::model_selection::train_test_split;
// Load dataset
let cancer_data = breast_cancer::load_dataset();
// Transform dataset into a NxM matrix
let x = DenseMatrix::from_array(
    cancer_data.num_samples,
    cancer_data.num_features,
    &cancer_data.data,
);
// These are our target class labels
let y = cancer_data.target;
// Split dataset into training/test (80%/20%)
let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2);
// Fit KNN classifier
let knn = KNNClassifier::fit(
    &x_train,
    &y_train,
    Distances::euclidian(),
    Default::default(),
);
let y_hat_knn = knn.predict(&x_test);    
// Calculate test error
println!("AUC: {}", roc_auc_score(&y_test, &y_hat_knn));
```

Default value of \\(K\\) is 3. If you want to change value of this and other parameters replace `Default::default()` with instance of [`KNNClassifierParameters`]({{site.api_base_url}}/neighbors/knn_classifier/struct.KNNClassifierParameters.html).

### Nearest Neighbors Regression

KNN Regressor, implemented in [`KNNClassifier`]({{site.api_base_url}}/neighbors/knn_regressor/struct.KNNRegressor.html) is very similar to KNN Classifier, the only difference is that returned value is a real value instead of class label. To fit `KNNRegressor` to [Boston Housing]({{site.api_base_url}}/dataset/boston/index.html) dataset:

```rust
use smartcore::dataset::*;
// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
// KNN
use smartcore::math::distance::Distances;
use smartcore::neighbors::knn_regressor::KNNRegressor;
// Model performance
use smartcore::metrics::mean_squared_error;
use smartcore::model_selection::train_test_split;
// Load dataset
let cancer_data = boston::load_dataset();
// Transform dataset into a NxM matrix
let x = DenseMatrix::from_array(
    cancer_data.num_samples,
    cancer_data.num_features,
    &cancer_data.data,
);
// These are our target class labels
let y = cancer_data.target;
// Split dataset into training/test (80%/20%)
let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2);
// Fit KNN regressor
let knn = KNNRegressor::fit(
    &x_train,
    &y_train,
    Distances::euclidian(),
    Default::default(),
);
let y_hat_knn = knn.predict(&x_test);
// Calculate test error
println!("MSE: {}", mean_squared_error(&y_test, &y_hat_knn));
```

As with KNN Classifier you can change value of k and other parameters by passing an instance of [`KNNRegressorParameters`]({{site.api_base_url}}/neighbors/knn_regressor/struct.KNNRegressorParameters.html) to `fit` function.

### Nearest Neighbor Algorithms

The computational complexity of KNN increases with the size of the training dataset. This is because every time prediction is made algorithm has to search through all stored samples to find K nearest neighbors. Efficient implementation of KNN requires special data structure, like [CoverTree](https://en.wikipedia.org/wiki/Cover_tree) to speed up look-up of nearest neighbors during prediction.

Cover Tree is the default algorithm for KNN regressor and classifier. Change value of `algorithm` field of the `KNNRegressorParameters` or `KNNClassifierParameters` if you want to switch to brute force search method.

#### Brute Force

The brute force nearest neighbor search is the simplest algorithm that calculates the distance from the query point to every other point in the dataset while maintaining a list of K nearest items in a [Binary Heap](https://en.wikipedia.org/wiki/Binary_heap#Search). This algorithms does not maintain any search data structure and results in \\(O(n)\\) search time, where \\(n\\) is number of samples. Brute force search algorithm is implemented in [LinearKNNSearch]({{site.api_base_url}}/algorithm/neighbour/linear_search/index.html).

#### Cover Tree

Although Brute Force algorithms is very simple approach it outperforms a lot of space partitioning approaches like [k-d tree](https://en.wikipedia.org/wiki/K-d_tree) on higher dimensional spaces. However, the brute-force approach quickly becomes infeasible as the dataset grows in size. To address inefficiencies of Brute Force other data structures are used that reduce the required number of distance calculations by efficiently encoding aggregate distance information for the sample.

A [Cover Tree]({{site.api_base_url}}/algorithm/neighbour/cover_tree/index.html) is a tree data structure used for the partitiong of metric spaces to speed up nearest neighbor operations. Cover trees are fast in practice and have great theoretical properties: 

* Construction: \\(O(c^6n\log n)\\)
* Query:  \\(O(c^{12}\log n)\\),
* The cover tree requires  \\(O(n)\\) space.

where \\(n\\) is number of samples in a dataset and \\(c\\) denotes the expansion constant.

### Distance Metrics

The choice of distance metric for KNN algorithm largely depends on properties of your data. If you don't know which distance to use go with Euclidean distance function or choose metric that gives you the best performance on a hold out test set. 
There are many other distance measures that can be used with KNN in *SmartCore*

{:.table .table-striped .table-bordered}
| Distance Metric | Parameters | Description |
|:-:|:-:|:-:|
| [Euclidian]({{site.api_base_url}}/math/distance/euclidian/index.html) |  | \\(\sqrt{\sum_{i=1}^n (x_i-y_i)^2}\\) |
| [Hamming]({{site.api_base_url}}/math/distance/hamming/index.html) |  | the number of places where \\( x \\) and \\( y \\) differ |
| [Mahalanobis]({{site.api_base_url}}/math/distance/mahalanobis/index.html) | \\(S\\), covariance matrix of the dataset | \\(\sqrt{(x - y)^TS^{-1}(x - y)}\\) |
| [Manhattan]({{site.api_base_url}}/math/distance/manhattan/index.html) |  | \\(\sum_{i=0}^n \lvert x_i - y_i \rvert\\) |
| [Minkowski]({{site.api_base_url}}/math/distance/minkowski/index.html) | p, distance order | \\(\left(\sum_{i=0}^n \lvert x_i - y_i \rvert^p\right)^{1/p}\\) |

## Linear Models

Linear regression is one of the most well known and well understood algorithms in statistics and machine learning. 

<figure class="image" align="center">
  <img src="/assets/imgs/simple_regression.svg" alt="Simple linear regression">
  <figcaption>Figure 1. Simple linear regression.</figcaption>
</figure>

The model describes the relationship between a dependent variable y (also called the response) as a function of one or more independent, or explanatory variables \\(X_i\\). The general equation for a linear model is:
\\[y = \beta_0 + \sum_{i=1}^n \beta_iX_i + \epsilon\\]

where the target value \\(y\\) is a linear combination of the features \\(X_i\\).

### Linear Regression

Use `fit` method of [`LinearRegression`]({{site.api_base_url}}/linear/linear_regression/index.html) to fit Ordinary Least Squares to your data. 

```rust
use smartcore::dataset::*;
// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
// Linear Regression
use smartcore::linear::linear_regression::LinearRegression;
// Model performance
use smartcore::metrics::{mean_squared_error, roc_auc_score};
use smartcore::model_selection::train_test_split;
// Load dataset
let cancer_data = boston::load_dataset();
// Transform dataset into a NxM matrix
let x = DenseMatrix::from_array(
    cancer_data.num_samples,
    cancer_data.num_features,
    &cancer_data.data,
);
// These are our target class labels
let y = cancer_data.target;
// Split dataset into training/test (80%/20%)
let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2);
// Fit Linear Regression
let lr = LinearRegression::fit(&x_train, &y_train, Default::default());
let y_hat_lr = lr.predict(&x_test);
// Calculate test error
println!("MSE Logistic Regression: {}", mean_squared_error(&y_test, &y_hat_lr));
```

By default, *SmartCore* uses [SVD Decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition) to find estimates of \\(\beta_i\\) that minimizes the sum of the squared residuals. While SVD Decomposition provides the most stable solution, you might decide to go with [QR Decomposition](https://en.wikipedia.org/wiki/QR_decomposition) since this approach is more computationally efficient than SVD Decomposition. For comparison, runtime complexity of SVD Decomposition is \\(O(mn^2 + n^3)\\) vs \\(O(mn^2 + n^3/3)\\) for QR decomposition, where \\(n\\) and \\(m\\) are dimentions of input matrix \\(X\\). Use `solver` attribute of the [`LinearRegressionParameters`]({{site.api_base_url}}/linear/linear_regression/struct.LinearRegressionParameters.html) to choose between decomposition methods.

### Logistic Regression

Logistic regression uses linear model to represent relashionship between dependent and explanatory variables. Unlike linear regression, output in logistic regression is modeled as a binary value (0 or 1) rather than a numeric value. to squish output between 0 and 1 [Sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) is used.

In *SmartCore* Logistic Regression is represented by [`LogisticRegression`]({{site.api_base_url}}/linear/logistic_regression/index.html) struct that has methods `fit` and `predict`. 

```rust
use smartcore::dataset::*;
// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
// Logistic Regression
use smartcore::linear::logistic_regression::LogisticRegression;
// Model performance
use smartcore::metrics::{mean_squared_error, roc_auc_score};
use smartcore::model_selection::train_test_split;
// Load dataset
let cancer_data = breast_cancer::load_dataset();
// Transform dataset into a NxM matrix
let x = DenseMatrix::from_array(
    cancer_data.num_samples,
    cancer_data.num_features,
    &cancer_data.data,
);
// These are our target class labels
let y = cancer_data.target;
// Split dataset into training/test (80%/20%)
let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2);
// Logistic Regression
let lr = LogisticRegression::fit(&x_train, &y_train);
let y_hat_lr = lr.predict(&x_test);
// Calculate test error
println!("AUC Logistic Regression: {}", roc_auc_score(&y_test, &y_hat_lr));
```

*SmartCore* uses [Limited-memory BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS) routine to find optimal combination of \\(\beta_i\\) parameters. 

## Decision Trees

regression

## Ensemble methods

regression