---
layout: manual
title:  "Getting Started"
---

# Getting Started

Welcome to *SmartCore*, the most advanced machine learning library in Rust! In *SmartCore* you will find implementation of the most common machine learning (ML) algorithms and tools:

* __Regression__: Linear Regression (OLS), Decision Tree Regressor, Random Forest Regressor, K Nearest Neighbors
* __Classification__: Logistic Regressor, Decision Tree Classifier, Random Forest Classifier, Supervised Nearest Neighbors (KNN)
* __Clustering__: K-Means
* __Matrix Decomposition__: PCA, LU, QR, SVD, EVD
* __Distance Metrics__: Euclidian, Minkowski, Manhattan, Hamming, Mahalanobis
* __Evaluation Metrics__: Accuracy, AUC, Recall, Precision, F1, Mean Absolute Error, Mean Squared Error, R2

All of these algorithms are implemented in Rust. 

Why another machine learning library for Rust, you might ask? At the time of writing of this guide there are at least three [general-purpose ML libraries](http://www.arewelearningyet.com/) for Rust.
Most of these libraries either do not support all of the algorithms that are implemented in *SmartCore* or aren't integrated with [nalgebra](https://nalgebra.org/) and [ndarray](https://github.com/rust-ndarray/ndarray).
While *SmartCore* is integrated with both libraries you can also use standard Rust data types, like vectors and arrays with all of the algorithms implemented here if you prefer not to depend on any linear algebra library.

We developed *SmartCore* to promote scientific computing in Rust. Our goal is to build an open library that has accurate, numerically stable, and well-documented implementations of the most well-known and widely used machine learning methods.

## Quick start

To start using *SmartCore* simply add the following to your `Cargo.toml` file:

```yaml
[dependencies]
smartcore = "0.1.0"
```

You will also have to decide which linear algebra library to use with *SmartCore*. We support:
* [ndarray](https://docs.rs/ndarray) - provides an n-dimensional container for general elements and for numerics.
* [nalgebra](https://docs.rs/nalgebra/) - general-purpose linear algebra for computer graphics and computer physics.

If you prefer not to depend on any external libraries other than *SmartCore* you can use all algorithms implemented here with Rust [Vec](https://doc.rust-lang.org/std/vec/struct.Vec.html) and [array](https://doc.rust-lang.org/std/primitive.array.html). 

### Iris Flower Classification

You are ready to fit your first ML algorithm to [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris). The Iris Dataset was introduced by the Ronald Fisher and is by far the best known database to be found in the pattern recognition literature. It consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters. It is often used in classification and clustering examples. First let's fit [K-Nearest Neighbors]({{site.api_base_url}}/neighbors/index.html) algorithm to Iris data:

```rust
use smartcore::dataset::iris::load_dataset;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::neighbors::knn_classifier::*;
use smartcore::math::distance::Distances;
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
// Fit KNN classifier to Iris dataset
let knn = KNNClassifier::fit(
    &x,
    &y,
    Distances::euclidian(), // We use euclidian distance here.
    Default::default(),
).unwrap();
let y_hat = knn.predict(&x).unwrap(); // Predict class labels
// Calculate training error
println!("accuracy: {}", accuracy(&y, &y_hat)); // Prints 0.96
```
Most of algorithms in *SmartCore* follow the same template when it comes to function names. There is usually a static function `fit` that fits model to your data and a function `predict` to predict labels or target values from new data. All mandatory parameters of the model are declared as parameters of function `fit`. All optional parameters are hidden behind `Default::default()`.

Now, if you want to fit [Logistic Regression]({{site.api_base_url}}/linear/logistic_regression/index.html) to your data you don't have to change your code a lot:

```rust
use smartcore::dataset::iris::load_dataset;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linear::logistic_regression::LogisticRegression;
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
let lr = LogisticRegression::fit(&x, &y).unwrap();
let y_hat = lr.predict(&x).unwrap(); // Predict class labels
// Calculate training error
println!("accuracy: {}", accuracy(&y, &y_hat)); // Prints 0.98
```

Our performance metric (accuracy) went up two percentage points! Nice work!

## High-level overview

Majority of machine learning algorithms rely on linear algebra and optimization methods at it's core. To make *SmartCore* as flexible and expandable as possible we decided not to build it on top of any specific linear algebra or optimization library. Instead, machine learning algorithms in *SmartCore* use an abstraction layer where operations on multidimensional arrays and maximization/minimization routines are defined. This allow us to quickly develop and plug-in a new type of matrix or vector as long as it implements all abstract methods from this layer. 

All routines from optimization module are not available directly but we plan to make optimization library public once it is mature enough. 

While functions from linear algebra module are public you should not use them directly because this module is still unstable and we plan to add and remove functions in the future. We opened these functions only to let anyone add implementations of other types of matrices that are currently not supported by *SmartCore*. Please see [Developer's Guide](/user_guide/developer.html) if you want to add new implementation for your favourite matrix type.

<figure class="image" align="center">
  <img src="/assets/imgs/architecture.svg" alt="SmartCore's architecture">
  <figcaption>Figure 1. SmartCore's architecture represented as layers.</figcaption>
</figure>

As you see from Figure 1 we have 3 layers with all the machine learning models defined at the second level. Model evaluation and selection deserves its own layer since functions defined here take models or predicted values as input.

### Input and Output

Most algorithms in *SmartCore* take 2D arrays (matrices) and vectors as input and produce matrices and vectors on output, as demonstrated in Figure 2.

<figure class="image" align="center">
  <img src="/assets/imgs/io_format.svg" alt="Input and output format.">
  <figcaption>Figure 2. Input and output as matrices and vectors where <em>N</em> is number of samples and <em>M</em> is number of features in each sample.</figcaption>
</figure>

**X** is a *NxM* matrix with your training data. Training data comes as a set of samples of length *N*. Each sample has *M* features. For supervised learning, you also provide a vector of training labels, **y** that should have length M. Traits [`BaseMatrix`]({{site.api_base_url}}/linalg/trait.BaseMatrix.html) and [`BaseVector`]({{site.api_base_url}}/linalg/trait.BaseVector.html) are where all matrix and vector operations used by *SmartCore* are defined. 

## Linear algebra libraries

All functions in *SmartCore* work well and thoroughly tested on simple Rust's vectors but we do recommend to use more advanced and faster crates for linear algebra such as [ndarray](https://docs.rs/ndarray) and [nalgebra](https://docs.rs/nalgebra/). To enable both libraries add these compilation features to your `Cargo.toml` file:

```yaml
[dependencies]
smartcore = { version = "0.1.0", features=["nalgebra-bindings", "ndarray-bindings"]}
```

Here is how you would fit Logistic Regression to Iris dataset loaded as `ndarray` matrix:

```rust
use smartcore::dataset::iris::load_dataset;
// ndarray
use ndarray::Array;
// Imports for Logistic Regression
use smartcore::linear::logistic_regression::LogisticRegression;
// Model performance
use smartcore::metrics::accuracy;
// Load Iris dataset
let iris_data = load_dataset();
// Turn Iris dataset into NxM matrix
let x = Array::from_shape_vec(
    (iris_data.num_samples, iris_data.num_features),
    iris_data.data,
).unwrap();
// These are our target class labels
let y = Array::from_shape_vec(iris_data.num_samples, iris_data.target).unwrap();
// Fit Logistic Regression to Iris dataset
let lr = LogisticRegression::fit(&x, &y).unwrap();
let y_hat = lr.predict(&x).unwrap(); // Predict class labels
// Calculate training error
println!("accuracy: {}", accuracy(&y, &y_hat)); // Prints 0.98
```

As you might have noticed already the only difference between this example and the previous are lines where `x` and `y` variables are defined. 

## What can I do next?

If you are done reading through this page we would recommend to go to a specific section that interests you most. User's manual is organized into these broad categories:
* [Supervised Learning](/user_guide/supervised.html), in this section you will find tree-based, linear and KNN models.
* [Unsupervised Learning](/user_guide/unsupervised.html), unsupervised methods like clustering and matrix decomposition methods.
* [Model Selection](/user_guide/model_selection.html), varios metrics for model evaluation.
* [Developer's Guide](/user_guide/developer.html), would you like to contribute? Here you will find useful guidelines and rubrics to consider.