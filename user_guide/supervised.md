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
// KNN classifier
let y_hat_knn = KNNClassifier::fit(
    &x_train,
    &y_train,
    Distances::euclidian(),
    Default::default(),
).and_then(|knn| knn.predict(&x_test)).unwrap();    
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
// KNN regressor
let y_hat_knn = KNNRegressor::fit(
    &x_train,
    &y_train,
    Distances::euclidian(),
    Default::default(),
).and_then(|knn| knn.predict(&x_test)).unwrap();
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
  <img src="/assets/imgs/simple_regression.svg" alt="Simple linear regression" class="img-fluid">
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
// Linear Regression
let y_hat_lr = LinearRegression::fit(&x_train, &y_train, Default::default())
    .and_then(|lr| lr.predict(&x_test)).unwrap();
// Calculate test error
println!("MSE: {}", mean_squared_error(&y_test, &y_hat_lr));
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
let y_hat_lr = LogisticRegression::fit(&x_train, &y_train)
    .and_then(|lr| lr.predict(&x_test)).unwrap();
// Calculate test error
println!("AUC: {}", roc_auc_score(&y_test, &y_hat_lr));
```

*SmartCore* uses [Limited-memory BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS) routine to find optimal combination of \\(\beta_i\\) parameters. 

## Decision Trees

Classification and Regression Trees (CART) and its modern variant Random Forest are among the most powerful algorithms available in machine learning. 

CART models relationship between predictor and explanatory variables as a binary tree. Each node of the tree represents a decision that is made based on an outcome of a single attribute.
The leaf nodes of the tree represent an outcome. To make a prediction we take the mean of the training observations belonging to the leaf node for regression and the mode of observations for classification.

Given a dataset with just three explanatory variables and a qualitative dependent variable the tree might look like an example below.

<figure class="image" align="center">
  <img src="/assets/imgs/tree.svg" alt="Decision Tree example" class="img-fluid">
  <figcaption>Figure 2. An example of Decision Tree where target is a class.</figcaption>
</figure>

CART model is simple and useful for interpretation. However, they typically are not competitive with the best supervised learning approaches, like Logistic and Linear Regression, especially when the response can be well approximated by a linear model. Tree-based method is also non-robust which means that a small change in the data can cause a large change in the final estimated tree. That's why it is a common practice to combine prediction from multiple trees in ensemble to estimate predicted values. 

In *SmartCore* both, decision and regression trees can be found in the [`tree`]({{site.api_base_url}}/tree/index.html) module. Use [`DecisionTreeClassifier`]({{site.api_base_url}}/tree/decision_tree_classifier/index.html) to fit decision tree and [`DecisionTreeRegressor`]({{site.api_base_url}}/tree/decision_tree_regressor/index.html) for regression. 

To fit [`DecisionTreeClassifier`]({{site.api_base_url}}/tree/decision_tree_classifier/index.html) to [Breast Cancer]({{site.api_base_url}}/dataset/breast_cancer/index.html) dataset:

```rust
use smartcore::dataset::*;
// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
// Tree
use smartcore::tree::decision_tree_classifier::DecisionTreeClassifier;
use smartcore::tree::decision_tree_regressor::DecisionTreeRegressor;
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
// Decision Tree
let y_hat_tree = DecisionTreeClassifier::fit(&x_train, &y_train, Default::default())
    .and_then(|tree| tree.predict(&x_test)).unwrap();
// Calculate test error
println!("AUC: {}", roc_auc_score(&y_test, &y_hat_tree));
```

Here we have used default parameter values but in practice you will almost always use [k-fold cross validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) or hold-out validation dataset to fine tune your parameter values. 

## Ensemble methods

In ensemble learning we combine predictions from multiple base models to reduce the variance of predictions and decrease generalization error. Base models are assumed to be independent from each other. [Bagging](https://en.wikipedia.org/wiki/Bootstrap_aggregating) is one of the most streightforward ways to reduce correlation between base models in the ensemble. It works by taking repeated samples from the same training data set. As a result we generate _K_ different training data sets (bootstraps) that overlap but are not the same. We then train our base model on the each bootstrapped training set and average predictions for regression or use majority voting scheme for classification. 

### Random Forest

Random forest is an extension of bagging that also randomly selects a subset of features when training a tree. This improvement decorrelated the trees and hence decreases prediction error even more. Random forests have proven effective on a wide range of different predictive modeling problems. 

Let's fit [Random Forest regressor]({{site.api_base_url}}/ensemble/random_forest_regressor/index.html) to Boston Housing dataset:

```rust
use smartcore::dataset::*;
// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
// Random Forest
use smartcore::ensemble::random_forest_regressor::RandomForestRegressor;
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
// Random Forest
let y_hat_rf = RandomForestRegressor::fit(&x_train, &y_train, Default::default())
    .and_then(|rf| rf.predict(&x_test)).unwrap();
// Calculate test error
println!("MSE: {}", mean_squared_error(&y_test, &y_hat_rf));
```

You should get lower mean squared error here when compared to other methods from this manual. This is because by default Random Forest fits 100 independent trees to different bootstrapped training sets and calculates target value by averaging predictions from these trees.

[Random Forest classifier]({{site.api_base_url}}/ensemble/random_forest_classifier/index.html) works in a similar manner. The only difference is that you prediction targets should be nominal or ordinal values (class label).

## References
* ["Nearest Neighbor Pattern Classification" Cover, T.M., IEEE Transactions on Information Theory (1967)](http://ssg.mit.edu/cal/abs/2000_spring/np_dens/classification/cover67.pdf)
* ["The Elements of Statistical Learning: Data Mining, Inference, and Prediction" Trevor et al., 2nd edition](https://web.stanford.edu/~hastie/ElemStatLearn/)
* ["An Introduction to Statistical Learning", James G., Witten D., Hastie T., Tibshirani R.](http://faculty.marshall.usc.edu/gareth-james/ISL/)
* ["The Art of Computer Programming" Knuth, D, Vol. 3, 2nd ed, Sorting and Searching, 1998](https://www-cs-faculty.stanford.edu/~knuth/taocp.html)
* ["Cover Trees for Nearest Neighbor" Beygelzimer et al., Proceedings of the 23rd international conference on Machine learning, ICML'06 (2006)](https://hunch.net/~jl/projects/cover_tree/cover_tree.html)
* ["Faster cover trees." Izbicki et al., Proceedings of the 32nd International Conference on Machine Learning, ICML'15 (2015)](http://www.cs.ucr.edu/~cshelton/papers/index.cgi%3FIzbShe15)
* ["Numerical Recipes: The Art of Scientific Computing",  Press W.H., Teukolsky S.A., Vetterling W.T, Flannery B.P, 3rd ed.](http://numerical.recipes/)
* ["Pattern Recognition and Machine Learning", C.M. Bishop, Linear Models for Classification](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
* ["On the Limited Memory Method for Large Scale Optimization", Nocedal et al., Mathematical Programming, 1989](http://users.iems.northwestern.edu/~nocedal/PDFfiles/limited.pdf)
* ["Classification and regression trees", Breiman, L, Friedman, J H, Olshen, R A, and Stone, C J, 1984](https://www.sciencebase.gov/catalog/item/545d07dfe4b0ba8303f728c1)