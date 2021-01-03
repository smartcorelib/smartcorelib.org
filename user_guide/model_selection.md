---
layout: manual
title:  "Model Selection and Evaluation"
---

# Model Selection and Evaluation

*SmartCore* comes with a lot of easy-to-use algorithms and it is straightforward to fit many different machine learning models to a given dataset. Once you have many algorithms to choose from the question becomes how to choose the best machine learning model among a range of different models that you can use for your data. The problem of choosing the right model becomes even harder if you consider many different combinations of hyperparameters for each algorithm.

Model selection is the process of selecting one final machine learning model from among a collection of candidate models for you problem at hand. The process of assessing a model’s performance is known as model evaluation.

[K-fold Cross-Validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) (k-fold CV) is a commonly used technique for model selection and evaluation. Another alternative is to split your data into three separate sets: _training_, _validation_, _test_. You use the _training_ set to train your model and _validation_ set for model selection and hyperparameter tuning. The _test_ set can be used to get an unbiased estimate of model performance.

You can use [`train_test_split`]({{site.api_base_url}}/model_selection/fn.train_test_split.html) function to split your data into two separate sets. In the example that follows we split [Boston Housing]({{site.api_base_url}}/dataset/boston/index.html) dataset into two new sets: 80% of data is reserved for training and 20% of data is left for model evaluation.

```rust
use smartcore::dataset::*;
// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
// Model performance
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
// Split data 80/20
let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
```

While a simple test/train split method is good method that works well with a very large dataset, the test score dependents on how the data is split into train and test sets. To get a better indication of how well your model performs on unseen data use k-fold CV. 

To evaluate performance of your model with k-fold CV use [`cross_validate`]({{site.api_base_url}}/model_selection/fn.cross_validate.html) function.
This function splits datasets up into k groups. One of the groups is used as the test set and the rest are used as the training set. The model is trained on the training set and scored on the test set. Then the process is repeated until each unique group as been used as the test set. 

For example, when you split your dataset into 3 folds, as in <nobr>Figure 1</nobr>, `cross_validate` will fit and evaluate your model 3 times. First, the function will use folds 2 and 3 to train your model and fold 1 to evaluate its performance. On the second run, the function will take folds 1 and 3 for trainig and fold 2 for evaluation. 

<figure class="image" align="center">
  <img src="{{site.baseurl}}/assets/imgs/kfold.svg" alt="k-fold CV" class="img-fluid">
  <figcaption>Figure 1. Illustration of the k-fold cross validation.</figcaption>
</figure>

Let's evaluate performance of Logistic Regression on Breast Cancer dataset with 3-fold CV:

```rust
use smartcore::dataset::breast_cancer;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
// Logistic regression
use smartcore::linear::logistic_regression::LogisticRegression;
// Model performance
use smartcore::metrics::accuracy;
// K-fold CV
use smartcore::model_selection::{cross_validate, KFold};
// Load dataset
let breast_cancer_data = breast_cancer::load_dataset();
let x = DenseMatrix::from_array(
    breast_cancer_data.num_samples,
    breast_cancer_data.num_features,
    &breast_cancer_data.data,
);
// These are our target values
let y = breast_cancer_data.target;
// cross-validated estimator
let results = cross_validate(
    LogisticRegression::fit,
    &x,
    &y,
    Default::default(),
    KFold::default().with_n_splits(3),
    accuracy,
)
.unwrap();
println!(
    "Test score: {}, training score: {}",
    results.mean_test_score(),
    results.mean_train_score()
);
```

Note that training score is higher than test score. The function [`cross_val_predict`]({{site.api_base_url}}/model_selection/fn.cross_val_predict.html) has a similar interface to `cross_validate`, but returns, for each element in the input, the prediction that was obtained for that element when it was in the test set.

## Toy Datasets

Toy datasets are part of *SmartCore* to give us an easy and fast way to demonstrate algorithms. We've intentionally selected small datasets because they are packaged and distributed with *SmartCore*. 
We also keep toy datasets behind the `datasets` feature flag. Feature `datasets` is included in `default`. If you want to exclude toy dataset you'll have to turn feature `default` off.

```yaml
[dependencies]
smartcore = { version = "0.1.0", default-features = false}
```

When feature flag `datasets` is enabled you'l get these datasets:

{:.table .table-striped .table-bordered}
| Dataset | Description | Samples | Attributes | Type |
|:-:|-|-|-|-|
| [The Boston Housing Dataset]({{site.api_base_url}}/dataset/boston/index.html) | The data  is derived from information collected by the U.S. Census Service concerning housing in the area of Boston, MA. | 506 | 13 | Regression |
| [Breast Cancer Wisconsin (Diagnostic) Data Set]({{site.api_base_url}}/dataset/breast_cancer/index.html) | Breast Cancer  was collected by Dr. William H. Wolberg, W. Nick Street and Olvi L. Mangasarian. Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass and describe characteristics of the cell nuclei present in the image. | 569 | 30 | Classification |
| [Diabetes Data]({{site.api_base_url}}/dataset/diabetes/index.html) | Diabetes Data  was collected by Bradley Efron, Trevor Hastie, Iain Johnstone and Robert Tibshirani for the "Least Angle Regression" paper. | 442 | 10 | Regression |
| [Digits]({{site.api_base_url}}/dataset/digits/index.html) | Optical Recognition of Handwritten Digits Data Set. The dataset  contains normalized bitmaps of handwritten digits (0-9) from a preprinted form. | 1797 | 64 | Classification, Clusteing |
| [The Iris Dataset flower]({{site.api_base_url}}/dataset/iris/index.html) | Fisher's Iris dataset  is a multivariate dataset that was published in 1936 by Ronald Fisher. | 150 | 4 | Classification |

## Metrics

There are three different groups of [evaluation metrics]({{site.api_base_url}}/metrics/index.html) in *SmartCore*: [classification metrics]({{site.api_base_url}}/metrics/struct.ClassificationMetrics.html), [regression metrics]({{site.api_base_url}}/metrics/struct.RegressionMetrics.html) and [clustering metrics]({{site.api_base_url}}/metrics/struct.ClusterMetrics.html). 

Most metrics evaluate instances of [`BaseVector`]({{site.api_base_url}}/linalg/trait.BaseVector.html) and thus work with any vectors that implement this trait. 

This is how you would calculate accuracy of your results if your estimated values are in `Vec<f32>`.

```rust
let y_pred: Vec<f32> = vec![0., 2., 1., 3.];
let y_true: Vec<f32> = vec![0., 1., 2., 3.];
let accuracy = smartcore::metrics::accuracy(&y_pred, &y_true);
```

## Model persistence

All algorithms and data structures in *SmartCore* implement `Deserialize` and `Serialize` traits from the [Serde](https://serde.rs/) crate. This enables you to serialize and deserialize your model into [any format supported by Serde](https://serde.rs/#data-formats). 

For example, to save your model on disk as a [bincode-encoded](https://github.com/servo/bincode) file make sure to add these lines into your `Cargo.toml` file.

```yaml
[dependencies]
serde = "1.0.115"
bincode = "1.3.1"
```

This code shows how you could save a KNN model on disk as a [bincode-encoded](https://github.com/servo/bincode) file and load it from the file later.

```rust
use std::fs::File;
use std::io::prelude::*;
use smartcore::dataset::iris::load_dataset;
// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
// Imports for KNN classifier
use smartcore::math::distance::*;
use smartcore::neighbors::knn_classifier::*;
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
// Fit KNN classifier to Iris dataset
let knn = KNNClassifier::fit(
    &x,
    &y,    
    Default::default(),
).unwrap();
// File name for the model
let file_name = "iris_knn.model";
// Save the model
{
    let knn_bytes = bincode::serialize(&knn).expect("Can not serialize the model");
    File::create(file_name)
        .and_then(|mut f| f.write_all(&knn_bytes))
        .expect("Can not persist model");
}
// Load the model
let knn: KNNClassifier<f32, euclidian::Euclidian> = {
    let mut buf: Vec<u8> = Vec::new();
    File::open(&file_name)
        .and_then(|mut f| f.read_to_end(&mut buf))
        .expect("Can not load model");
    bincode::deserialize(&buf).expect("Can not deserialize the model")
};
//Predict class labels
let y_hat = knn.predict(&x).unwrap(); // Predict class labels
// Calculate training error
println!("accuracy: {}", accuracy(&y, &y_hat)); // Prints 0.96
```