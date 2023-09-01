---
layout: manual
title:  Supervised Learning
description: Supervised learning with Smartcore, including, but not limited to KNN, Naive Bayes, Decision Trees, ensemble methods, Random Forest, linear models and SVM.
---

# Supervised Learning

Most machine learning problems fall into one of two categories: _supervised_ and _unsupervised_.
This page describes supervised learning algorithms implemented in *SmartCore*. 

In supervised learning we build a model which can be written in the very general form as
\\[Y = f(X) + \epsilon\\]
\\(X = (X_1,X_2,...,X_p)\\) are observations that consist of \\(p\\) different predictors, \\(Y\\) is an associated target value and 
\\(\epsilon\\) is a random error term, which is independent of \\(X\\) and has zero mean.
We fit an unknown function \\(f\\) to our data to predict the response for future observations or better understand the relationship between the response and the predictors.

Supervised learning is by far the most common machine learning method that is used in practice. Supervised learning problems can be grouped into regression and classification:

* *Classification*: when the output variable is qualitative (category), such as "cancerous" or "benign", "round" or "square".
* *Regression*: a regression problem is when the output variable is quantitative, such as "height" or "dollars".

All algorithms in *SmartCore* support both, qualitative and quantitative response variables and follow the same naming convention for function names. 
To fit an algorithm to your data use `fit` method that takes 2 mandatory parameters: `x` for your predictors and `y` for target values. All optional parameters are hidden behind `Default::default()`.
To make a prediction use `predict` method that takes new observations as `x` and predicts estimated class labels or target values. 

## K Nearest Neighbors 

K-nearest neighbors (KNN) is one of the simplest and best-known non-parametric classification and regression methods.
KNN does not require training. The algorithm simply stores the entire dataset and then uses this dataset to make predictions.

More formally,
given a positive integer \\(K\\) and a test observation \\(x_0\\), the KNN classifier first identifies the \\(K\\) points in the training data that are closest to \\(x_0\\), represented by \\(N_0\\) and then estimates the conditional probability for class \\(j\\) as the fraction of points in N0 whose response values equal \\(j\\):

\\[ Pr(Y=j \vert X=x_0) = \frac{1}{K} \sum_{i \in N_0} I(y_i=j) \\]

KNN Regressor is closely related to the KNN classifier. It estimates a target value using the average of all the reponses in \\(N_0\\), i.e.

\\[ \hat{y} = \frac{1}{K} \sum_{i \in N_0} y_i \\]

The choice of \\(K\\) is very important. \\(K\\) can be found by tuning the algorithm on a holdout dataset. It is a good idea to try many different values for \\(K\\) (e.g. values from 1 to 21) and see which value gives the best test error rate.

To determine which of the \\(K\\) instances in the training dataset are most similar to a new input a [distance metric]({{site.api_base_url}}/math/distance/index.html) is used. 
For real-valued input variables, the most popular distance metric is [Euclidean distance]({{site.api_base_url}}/math/distance/euclidian/index.html). You can choose the best distance metric based on the properties of your data. If you are unsure, you can experiment with different distance metrics and different values of \\(K\\) together and see which mix results in the most accurate models.

### Nearest Neighbors Classification 

To fit KNN Classifier to your data use [`KNNClassifier`]({{site.api_base_url}}/neighbors/knn_classifier/struct.KNNClassifier.html). Let's fit KNN Classifier to the [Breast Cancer]({{site.api_base_url}}/dataset/breast_cancer/index.html) dataset:

```rust
use smartcore::dataset::*;
// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
// Imports for KNN classifier
use smartcore::neighbors::knn_classifier::KNNClassifier;
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
let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
// KNN classifier
let y_hat_knn = KNNClassifier::fit(
    &x_train,
    &y_train,        
    Default::default(),
).and_then(|knn| knn.predict(&x_test)).unwrap();    
// Calculate test error
println!("AUC: {}", roc_auc_score(&y_test, &y_hat_knn));
```

Default value of \\(K\\) is 3. If you want to change this value or other parameters replace `Default::default()` with an instance of [`KNNClassifierParameters`]({{site.api_base_url}}/neighbors/knn_classifier/struct.KNNClassifierParameters.html).

### Nearest Neighbors Regression

KNN Regressor, implemented in [`KNNClassifier`]({{site.api_base_url}}/neighbors/knn_regressor/struct.KNNRegressor.html) is very similar to KNN Classifier, the only difference is that the returned value is a real value instead of class label. To fit `KNNRegressor` to [Boston Housing]({{site.api_base_url}}/dataset/boston/index.html) dataset:

```rust
use smartcore::dataset::*;
// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
// KNN
use smartcore::math::distance::Distances;
use smartcore::neighbors::knn_regressor::{KNNRegressor, KNNRegressorParameters};
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
let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
// KNN regressor
let y_hat_knn = KNNRegressor::fit(
    &x_train,
    &y_train,    
    KNNRegressorParameters::default().with_distance(Distances::euclidian()),
).and_then(|knn| knn.predict(&x_test)).unwrap();
// Calculate test error
println!("MSE: {}", mean_squared_error(&y_test, &y_hat_knn));
```

As with KNN Classifier you can change value of k and other parameters by passing an instance of [`KNNRegressorParameters`]({{site.api_base_url}}/neighbors/knn_regressor/struct.KNNRegressorParameters.html) to `fit` function.

### Nearest Neighbor Algorithms

The computational complexity of KNN increases with the size of the training dataset. This is because every time a prediction is made the algorithm has to search through all stored samples to find K nearest neighbors. Efficient implementations of KNN require a special data structure, like [CoverTree](https://en.wikipedia.org/wiki/Cover_tree) to speed up the look-up of nearest neighbors during prediction.

Cover Tree is the default algorithm for KNN regressor and classifier. Change the value of `algorithm` field of the `KNNRegressorParameters` or `KNNClassifierParameters` if you want to switch to brute force search method.

#### Brute Force

The brute force nearest neighbor search is the simplest algorithm that calculates the distance from the query point to every other point in the dataset while maintaining a list of K nearest items in a [Binary Heap](https://en.wikipedia.org/wiki/Binary_heap#Search). This algorithm does not maintain any search data structure and results in \\(O(n)\\) search time, where \\(n\\) is number of samples. Brute force search algorithm is implemented in [LinearKNNSearch]({{site.api_base_url}}/algorithm/neighbour/linear_search/index.html).

#### Cover Tree

Although the Brute Force algorithm is a very simple approach it outperforms a lot of space partitioning approaches like [k-d tree](https://en.wikipedia.org/wiki/K-d_tree) on higher dimensional spaces. However, the brute-force approach quickly becomes infeasible as the dataset grows in size. To address inefficiencies of Brute Force other data structures are used that reduce the required number of distance calculations by efficiently encoding aggregate distance information for the sample.

A [Cover Tree]({{site.api_base_url}}/algorithm/neighbour/cover_tree/index.html) is a tree data structure used for the partitioning of metric spaces to speed up nearest neighbor operations. Cover trees are fast in practice and have great theoretical properties: 

* Construction: \\(O(c^6n\log n)\\)
* Query:  \\(O(c^{12}\log n)\\),
* The cover tree requires  \\(O(n)\\) space.

where \\(n\\) is number of samples in a dataset and \\(c\\) denotes the expansion constant.

### Distance Metrics

The choice of distance metric for the KNN algorithm largely depends on properties of your data. If you don't know which distance to use go with Euclidean distance function or choose a metric that gives you the best performance on a hold out test set. 
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
  <img src="{{site.baseurl}}/assets/imgs/simple_regression.svg" alt="Simple linear regression" class="img-fluid">
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
let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
// Linear Regression
let y_hat_lr = LinearRegression::fit(&x_train, &y_train, Default::default())
    .and_then(|lr| lr.predict(&x_test)).unwrap();
// Calculate test error
println!("MSE: {}", mean_squared_error(&y_test, &y_hat_lr));
```

By default, *SmartCore* uses [SVD Decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition) to find estimates of \\(\beta_i\\) that minimize the sum of the squared residuals. While the SVD provides the most stable solution, you might decide to go with [QR Decomposition](https://en.wikipedia.org/wiki/QR_decomposition) since this approach is more computationally efficient than the SVD. For comparison, runtime complexity of the SVD is \\(O(mn^2 + n^3)\\) vs \\(O(mn^2 + n^3/3)\\) for QR decomposition, where \\(n\\) and \\(m\\) are dimentions of input matrix \\(X\\). Use `solver` attribute of the [`LinearRegressionParameters`]({{site.api_base_url}}/linear/linear_regression/struct.LinearRegressionParameters.html) to choose between decomposition methods.

### Shrinkage Methods

One way to avoid overfitting when you fit a linear model to your dataset is to use regularization. In simple terms, regularization shrinks parameters of the model towards zero. This shrinkage has the effect of reducing variance. Depending on what type of shrinkage is performed, some of the coefficients may be estimated to be exactly zero. Hence, shrinkage methods can also perform variable selection.

#### Ridge Regression

Ridge Regression is a regularized version of linear regression that adds an L2 regularization term to the cost function:

\\[\lambda \sum_{i=i}^n \beta_i^2\\] 

where \\(\lambda \geq 0\\) is a tuning hyperparameter. If \\(\lambda\\) is close to 0, then it has no effect because Ridge Regression is similar to plain linear regression. As \\(\lambda\\) gets larger the shrinking effect on the weights gets stronger and the weights approach zero.

To fit Ridge Regression use structs from the [`ridge_regression`]({{site.api_base_url}}/linear/ridge_regression/index.html) module:

```rust
use smartcore::dataset::*;
// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linear::ridge_regression::{RidgeRegression, RidgeRegressionParameters};
// Model performance
use smartcore::metrics::mean_squared_error;
use smartcore::model_selection::train_test_split;
// Load dataset
let boston_data = boston::load_dataset();
// Transform dataset into a NxM matrix
let x = DenseMatrix::from_array(
    boston_data.num_samples,
    boston_data.num_features,
    &boston_data.data,
);
// These are our target values
let y = boston_data.target;
let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
// Ridge Regression
let y_hat_rr = RidgeRegression::fit(
    &x_train,
    &y_train,
    RidgeRegressionParameters::default().with_alpha(0.5),
)
.and_then(|rr| rr.predict(&x_test))
.unwrap();
// Calculate test error
println!(
    "MSE Ridge Regression: {}",
    mean_squared_error(&y_test, &y_hat_rr)
);
```

#### LASSO

LASSO stands for Least Absolute Shrinkage and Selection Operator. It is analogous to Ridge Regression but uses an L1 regularization term instead of an L2 regularization term:

\\[\lambda \sum_{i=i}^n \mid \beta_i \mid \\] 

As with ridge regression, the lasso shrinks the coefficient estimates towards zero. However, in the case of the lasso, the L1 penalty has the effect of forcing some of the coefficient estimates to be exactly equal to zero when the tuning parameter \\(\lambda\\) is sufficiently large. Hence, the lasso performs variable selection and models generated from the lasso are generally much easier to interpret than those produced by ridge regression.

```rust
use smartcore::dataset::*;
// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linear::lasso::{Lasso, LassoParameters};
// Model performance
use smartcore::metrics::mean_squared_error;
use smartcore::model_selection::train_test_split;
// Load dataset
let boston_data = boston::load_dataset();
// Transform dataset into a NxM matrix
let x = DenseMatrix::from_array(
    boston_data.num_samples,
    boston_data.num_features,
    &boston_data.data,
);
// These are our target values
let y = boston_data.target;
let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
// LASSO
let y_hat_lasso = Lasso::fit(
    &x_train,
    &y_train,
    LassoParameters::default().with_alpha(0.5),
)
.and_then(|lr| lr.predict(&x_test))
.unwrap();
// Calculate test error
println!("MSE LASSO: {}", mean_squared_error(&y_test, &y_hat_lasso));
```

#### Elastic Net

Elastic net linear regression uses the penalties from both the lasso and ridge techniques to regularize regression models.

\\[\lambda_1 \sum_{i=i}^n \beta_i^2 + \lambda_2 \sum_{i=i}^n \mid \beta_i \mid\\]

where \\(\lambda_1 = \\alpha l_{1r}\\), \\(\lambda_2 = \\alpha (1 -  l_{1r})\\) and \\(l_{1r}\\) is the l1 ratio, elastic net mixing parameter.

Elastic net combines both the L1 and L2 penalties during training, which can result in better performance than a model with either one or the other penalty on some problems.

```rust
use smartcore::dataset::*;
// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linear::elastic_net::{ElasticNet, ElasticNetParameters};
// Model performance
use smartcore::metrics::mean_squared_error;
use smartcore::model_selection::train_test_split;
// Load dataset
let boston_data = boston::load_dataset();
// Transform dataset into a NxM matrix
let x = DenseMatrix::from_array(
    boston_data.num_samples,
    boston_data.num_features,
    &boston_data.data,
);
// These are our target values
let y = boston_data.target;
let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
// Elastic Net
let y_hat_en = ElasticNet::fit(
    &x_train,
    &y_train,
    ElasticNetParameters::default()
        .with_alpha(0.5)
        .with_l1_ratio(0.5),
)
.and_then(|lr| lr.predict(&x_test))
.unwrap();
// Calculate test error
println!(
    "MSE Elastic Net: {}",
    mean_squared_error(&y_test, &y_hat_en)
);
```

### Logistic Regression

Logistic regression uses a linear model to represent a relationship between dependent and explanatory variables. Unlike linear regression, output in logistic regression is modeled as a binary value (0 or 1) rather than a numeric value. to squish output between 0 and 1 [Sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) is used.

In *SmartCore* Logistic Regression is represented by [`LogisticRegression`]({{site.api_base_url}}/linear/logistic_regression/index.html) struct that has methods `fit` and `predict`. 

```rust
use smartcore::dataset::*;
// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
// Logistic Regression
use smartcore::linear::logistic_regression::LogisticRegression;
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
let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
// Logistic Regression
let y_hat_lr = LogisticRegression::fit(&x_train, &y_train, Default::default())
    .and_then(|lr| lr.predict(&x_test)).unwrap();
// Calculate test error
println!("AUC: {}", roc_auc_score(&y_test, &y_hat_lr));
```

*SmartCore* uses [Limited-memory BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS) routine to find an optimal combination of \\(\beta_i\\) parameters. 

## Support Vector Machines

Support Vector Machines (SVM) are perhaps one of the most popular machine learning algorithms. SVMs have been shown to perform well in a variety of settings, and are often considered one of the best "out of the box" classifiers. The support vector machines is a generalization of a simple and intuitive classifier called the maximal margin classifier.

The maximal margin classifier is a hypothetical classifier that best explains how SVM works in practice. This classifier is based on the idea of a hyperplane, a flat affine subspace of dimension \\(p-1\\) that divides p-dimensional space into two halves. A hyperplane is defined by the equation

\\[\beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_pX_p = 0\\]

To classify a new point using this line we plug in input values into this equation, and check on which side of the hyperplane a point lies by calculating the sign of the left hand side of it. 
When the equation returns a value greater than 0 the point belongs to the first class, when the equation returns a value less than 0 and the point belongs to the second class.

The distance between the hyperplane and the closest data points is referred to as the margin. The best or optimal hyperplane that can separate the two classes is the hyperplane that has the largest margin.
This is called the maximal margin hyperplane.

In practice, real data is messy and cannot be separated perfectly with a hyperplane. The generalization of the maximal margin classifier to the non-separable case is known as the support vector classifier.

The support vector classifier (SVC) is an extension of the maximal margin classifier that results from enlarging the feature space in a specific way, using kernels. SVC allow some observations to be on the incorrect side of the margin, or even the incorrect side of the hyperplane rather than seeking the largest possible margin so that every observation is on the correct side of the hyperplane. 

<figure class="image" align="center">
  <img src="{{site.baseurl}}/assets/imgs/linear_svm.svg" alt="Simple linear regression" class="img-fluid">
  <figcaption>Figure 2. Linear decision boundary of the Support Vector Classifier.</figcaption>
</figure>

As maximal margin classifier, the SVC classifies a test observation depending on which side of a hyperplane it lies. The hyperplane is chosen to correctly separate most of the training observations into the two classes, but may misclassify a few observations. It is the solution to the optimization problem:

\\[\underset{\beta_1, \beta_2, ..., \beta_p, \epsilon_1, ..., \epsilon_n, M}{maximize} \space \space M \\]

subject to:
\\[\sum_{j=1}^p\beta_j^2  = 1\\]
\\[y_i(\beta_0 + \beta_1x_{i1} + \beta_2x_{i2} + ... + \beta_px_{ip}) \geq M(1 - \epsilon_i) \\]
\\[\epsilon_i \geq 0, \sum_{i=1}^n \epsilon_i \leq C\\]

where C is a nonnegative tuning parameter, M is the width of the margin and \\(\epsilon_1, ..., \epsilon_n\\) are slack variables that allow individual observations to be on the wrong side of the margin or the hyperplane.

C controls the bias-variance trade-off of the support vector classifier. When the tuning parameter C is large, then the margin is wide, many observations violate the margin, and so there are many support vectors. If C is small, then there will be fewer support vectors and hence the resulting classifier will have low bias but high variance.

The solution to the support vector classifier optimization problem involves only the inner products of the observations, rather than the observations themselves. When we replace the inner product with a generalization that defines the similarity between new data and the support vectors the resulting classifier is known as a support vector machine. The similarity between a data point and the support vectors is called the kernel function.

<figure class="image" align="center">
  <img src="{{site.baseurl}}/assets/imgs/rbf_svm.svg" alt="Simple linear regression" class="img-fluid">
  <figcaption>Figure 3. Support Vector Classifier with RBF kernel.</figcaption>
</figure>

*SmartCore* supports multiple kernel functions but you can always define a new kernel function by implementing the [`Kernel`]({{site.api_base_url}}/svm/trait.Kernel.html) trait. Not all functions can be a kernel.
Building a new kernel requires a good mathematical understanding of the [Mercer theorem](https://en.wikipedia.org/wiki/Mercer%27s_theorem)
that gives necessary and sufficient condition for a function to be a kernel function.

Pre-defined kernel functions:

{:.table .table-striped .table-bordered}
| Kernel | Description |
|-|-|
| Linear | \\( K(x, x') = \langle x, x' \rangle\\) |
| Polynomial | \\( K(x, x') = (\gamma\langle x, x' \rangle + r)^d\\), where \\(d\\) is polynomial degree, \\(\gamma\\) is a kernel coefficient and \\(r\\) is an independent term in the kernel function. |
| RBF (Gaussian) | \\( K(x, x') = e^{-\gamma \lVert x - x' \rVert ^2} \\), where \\(\gamma\\) is kernel coefficient |
| Sigmoid (hyperbolic tangent) | \\( K(x, x') = \tanh ( \gamma \langle x, x' \rangle + r ) \\), where \\(\gamma\\) is kernel coefficient and \\(r\\) is an independent term in the kernel function. |

### Support Vector Classifier

To fit a support vector classifier to your dataset use [`SVC`]({{site.api_base_url}}/svm/svc/index.html). 
*SmartCore* uses an [approximate SVM solver](https://leon.bottou.org/projects/lasvm) to solve the SVM optimization problem. 
The solver reaches accuracies similar to that of a real SVM after performing two passes through the training examples. 
You can choose the number of passes through the data that the algorithm takes by changing the `epoch` parameter of the classifier.

```rust
use smartcore::dataset::*;
// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
// SVM
use smartcore::svm::svc::{SVCParameters, SVC};
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
let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
// SVC
let y_hat_svm = SVC::fit(&x_train, &y_train, SVCParameters::default().with_c(10.0))
    .and_then(|svm| svm.predict(&x_test))
    .unwrap();
// Calculate test error    
println!("AUC SVM: {}", roc_auc_score(&y_test, &y_hat_svm));
```

### Support Vector Regressor

To fit support vector regressor to your dataset use [`epsilon-support SVR`]({{site.api_base_url}}/svm/svr/index.html).

```rust
use smartcore::dataset::*;
// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
// SVM
use smartcore::svm::svr::{SVRParameters, SVR};
use smartcore::svm::Kernels;
// Model performance
use smartcore::model_selection::train_test_split;
use smartcore::metrics::mean_squared_error;
// Load dataset
let diabetes_data = diabetes::load_dataset();
// Transform dataset into a NxM matrix
let x = DenseMatrix::from_array(
    diabetes_data.num_samples,
    diabetes_data.num_features,
    &diabetes_data.data,
);
// These are our target values
let y = diabetes_data.target;
// Split dataset into training/test (80%/20%)
let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
// SVM
let y_hat_svm = SVR::fit(&x_train, &y_train, 
    SVRParameters::default().with_kernel(Kernels::rbf(0.5)).
        with_c(2000.0).with_eps(10.0))
    .and_then(|svm| svm.predict(&x_test))
    .unwrap();
// Calculate test error
println!(
    "MSE: {}",
    mean_squared_error(&y_test, &y_hat_svm)
);
```

## Naive Bayes

Naive Bayes (NB) is a probabilistic machine learning algorithm based on the Bayes Theorem that assumes conditional independence between features given the value of the class variable.

Bayes Theorem states the following relationship between class label and data:

\\[ P(y \mid X) = \frac{P(y)P(X \mid y)}{P(X)} \\]

where
* \\(X = (x_1,...x_n)\\) represents the predictors.
* \\(P(y \mid X)\\) is the probability of class _y_ given the data X
* \\(P(X \mid y)\\) is the probability of data X given the class _y_.
* \\(P(y)\\) is the probability of class y. This is called the prior probability of y.
* \\(P(y \mid X)\\) is the probability of the data (regardless of the class value).

We are interested in calculating the posterior probability of \\(P(y \mid X)\\) from the prior probability \\(P(y)\\), conditional probability \\(P(X \mid y)\\) and \\(P(X)\\). 
The naive conditional independence assumption let us rewrite this equation as

\\[ P(y \mid x_1,...x_n) = \frac{P(y)\prod_{i=1}^nP(x_i \mid y)}{P(x_1,...x_n)} \\]

The denominator can be removed since \\(P(x_1,...x_n)\\) is constant for all the entries in the dataset.

\\[ P(y \mid x_1,...x_n) \propto P(y)\prod_{i=1}^nP(x_i \mid y) \\]

To find class y from predictors X we use this equation

\\[ y = \underset{y}{argmax} P(y)\prod_{i=1}^nP(x_i \mid y) \\]

Specific variants of the Naive Bayes classifier have different assumptions regarding the distribution of \\(P(x_i \mid y)\\). 
The table below displays variants of Naive Bayes classifiers implemented in *SmartCore*:

{:.table .table-striped .table-bordered}
| Name | The assumptions on distribution of features |
|-|-|
| [Bernoulli NB]({{site.api_base_url}}/naive_bayes/bernoulli/index.html) | features are independent binary variables |
| [Multinomial NB]({{site.api_base_url}}/naive_bayes/multinomial/index.html) | features are the frequencies with which certain events have been generated by a multinomial distribution |
| [Categorical NB]({{site.api_base_url}}/naive_bayes/categorical/index.html) | each feature has its own categorical distribution |
| [Gaussian NB]({{site.api_base_url}}/naive_bayes/gaussian/index.html) | continuous data distributed according to a Gaussian distribution |

For example, this is how you can fit Gaussian NB to the Iris dataset:

```rust
use smartcore::dataset::iris::load_dataset;
// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
// Imports Gaussian Naive Bayes classifier
use smartcore::naive_bayes::gaussian::GaussianNB;
// Model performance
use smartcore::metrics::accuracy;
// Load Iris dataset
let iris_data = load_dataset();
// Turn Iris dataset into NxM matrix
let x = DenseMatrix::from_array(
    iris_data.num_samples,
    iris_data.num_features,
    &iris_data.data,
);
// These are our target class labels
let y = iris_data.target;
// Fit Logistic Regression to Iris dataset
let gnb = GaussianNB::fit(&x, &y, Default::default()).unwrap();
let y_hat = gnb.predict(&x).unwrap(); // Predict class labels
// Calculate training error
println!("accuracy: {}", accuracy(&y, &y_hat)); // Prints 0.96
```

## Decision Trees

Classification and Regression Trees (CART) and its modern variant Random Forest are among the most powerful algorithms available in machine learning. 

CART models relationships between predictor and explanatory variables as a binary tree. Each node of the tree represents a decision that is made based on an outcome of a single attribute.
The leaf nodes of the tree represent an outcome. To make a prediction we take the mean of the training observations belonging to the leaf node for regression and the mode of observations for classification.

Given a dataset with just three explanatory variables and a qualitative dependent variable the tree might look like an example below.

<figure class="image" align="center">
  <img src="{{site.baseurl}}/assets/imgs/tree.svg" alt="Decision Tree example" class="img-fluid">
  <figcaption>Figure 4. An example of Decision Tree where target is a class.</figcaption>
</figure>

CART model is simple and useful for interpretation. However, they typically are not competitive with the best supervised learning approaches, like Logistic and Linear Regression, especially when the response can be well approximated by a linear model. Tree-based methods are also non-robust which means that a small change in the data can cause a large change in the final estimated tree. That's why it is a common practice to combine predictions from multiple trees in ensemble to estimate predicted values. 

In *SmartCore* both, decision and regression trees can be found in the [`tree`]({{site.api_base_url}}/tree/index.html) module. Use [`DecisionTreeClassifier`]({{site.api_base_url}}/tree/decision_tree_classifier/index.html) to fit decision tree and [`DecisionTreeRegressor`]({{site.api_base_url}}/tree/decision_tree_regressor/index.html) for regression. 

To fit [`DecisionTreeClassifier`]({{site.api_base_url}}/tree/decision_tree_classifier/index.html) to [Breast Cancer]({{site.api_base_url}}/dataset/breast_cancer/index.html) dataset:

```rust
use smartcore::dataset::*;
// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
// Tree
use smartcore::tree::decision_tree_classifier::DecisionTreeClassifier;
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
let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
// Decision Tree
let y_hat_tree = DecisionTreeClassifier::fit(&x_train, &y_train, Default::default())
    .and_then(|tree| tree.predict(&x_test)).unwrap();
// Calculate test error
println!("AUC: {}", roc_auc_score(&y_test, &y_hat_tree));
```

Here we have used default parameter values but in practice you will almost always use [k-fold cross validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) or hold-out validation dataset to fine tune your parameter values. 

## Ensemble methods

In ensemble learning we combine predictions from multiple base models to reduce the variance of predictions and decrease generalization error. Base models are assumed to be independent from each other. [Bagging](https://en.wikipedia.org/wiki/Bootstrap_aggregating) is one of the most straightforward ways to reduce correlation between base models in the ensemble. It works by taking repeated samples from the same training data set. As a result we generate _K_ different training data sets (bootstraps) that overlap but are not the same. We then train our base model on each bootstrapped training set and average predictions for regression or use majority voting scheme for classification. 

### Random Forest

Random forest is an extension of bagging that also randomly selects a subset of features when training a tree. This improvement decorrelates the trees and hence decreases prediction error even more. Random forests have proven effective on a wide range of different predictive modeling problems. 

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
let boston_data = boston::load_dataset();
// Transform dataset into a NxM matrix
let x = DenseMatrix::from_array(
    boston_data.num_samples,
    boston_data.num_features,
    &boston_data.data,
);
// These are our target class labels
let y = boston_data.target;
// Split dataset into training/test (80%/20%)
let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
// Random Forest
let y_hat_rf = RandomForestRegressor::fit(&x_train, &y_train, Default::default())
    .and_then(|rf| rf.predict(&x_test)).unwrap();
// Calculate test error
println!("MSE: {}", mean_squared_error(&y_test, &y_hat_rf));
```

You should get lower mean squared error here when compared to other methods from this manual. This is because by default Random Forest fits 100 independent trees to different bootstrapped training sets and calculates target value by averaging predictions from these trees.

[Random Forest classifier]({{site.api_base_url}}/ensemble/random_forest_classifier/index.html) works in a similar manner. The only difference is that your prediction targets should be nominal or ordinal values (class label).

## References
* ["Nearest Neighbor Pattern Classification" Cover, T.M., IEEE Transactions on Information Theory (1967)](http://ssg.mit.edu/cal/abs/2000_spring/np_dens/classification/cover67.pdf)
* ["The Elements of Statistical Learning: Data Mining, Inference, and Prediction" Trevor et al., 2nd edition](https://web.stanford.edu/~hastie/ElemStatLearn/)
* ["An Introduction to Statistical Learning", James G., Witten D., Hastie T., Tibshirani R.](http://faculty.marshall.usc.edu/gareth-james/ISL/)
* ["The Art of Computer Programming" Knuth, D, Vol. 3, 2nd ed, Sorting and Searching, 1998](https://www-cs-faculty.stanford.edu/~knuth/taocp.html)
* ["Machine Learning: A Probabilistic Perspective", Kevin P. Murphy, 2012, Chapter 3 ](https://mitpress.mit.edu/books/machine-learning-1)
* ["Cover Trees for Nearest Neighbor" Beygelzimer et al., Proceedings of the 23rd international conference on Machine learning, ICML'06 (2006)](https://hunch.net/~jl/projects/cover_tree/cover_tree.html)
* ["Faster cover trees." Izbicki et al., Proceedings of the 32nd International Conference on Machine Learning, ICML'15 (2015)](http://www.cs.ucr.edu/~cshelton/papers/index.cgi%3FIzbShe15)
* ["Numerical Recipes: The Art of Scientific Computing",  Press W.H., Teukolsky S.A., Vetterling W.T, Flannery B.P, 3rd ed.](http://numerical.recipes/)
* ["Pattern Recognition and Machine Learning", C.M. Bishop, Linear Models for Classification](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
* ["On the Limited Memory Method for Large Scale Optimization", Nocedal et al., Mathematical Programming, 1989](http://users.iems.northwestern.edu/~nocedal/PDFfiles/limited.pdf)
* ["Classification and regression trees", Breiman, L, Friedman, J H, Olshen, R A, and Stone, C J, 1984](https://www.sciencebase.gov/catalog/item/545d07dfe4b0ba8303f728c1)
* ["Support Vector Machines", Kowalczyk A., 2017](https://www.svm-tutorial.com/2017/10/support-vector-machines-succinctly-released/)
