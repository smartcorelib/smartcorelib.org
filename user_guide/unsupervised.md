---
layout: manual
title:  "Unsupervised Learning"
description: Unsupervised learning with Smartcore, including, but not limited to K-means, DBSCAN, SVD, PCA. Various matrix decomposition methods, like LU, EVD, QR
---

# Unsupervised Learning

In unsupervised learning we do not have a labeled dataset. In other words, for every observation \\(i = 1,...,n\\), we observe a vector of measurements \\(x_i\\) but no associated response \\(y_i\\). The goal for unsupervised learning is to model the underlying structure or distribution in the data in order to learn more about the problem at hand. 

In *SmartCore*, we use the same set of functions to fit unsupervised algorithms to your data as in supervised learning. The only difference is that the method `fit` does not need labels to learn from your data. Similar to supervised learning, optional parameters of method `fit` are hidden behind `Default::default()`. To make predictions use `predict` method that takes new data and predicts estimated class labels.

## Clustering

In clustering, we want to discover inherent homogeneous subgroups or clusters in our data. Unlike supervised learning, clustering algorithms use only measurements, \\(x_i\\) to automatically find natural subgroups in feature space.

A cluster is often an area of density in the feature space where observations are closer to each other than to other clusters. The cluster may have a center, a point in a feature space, and a boundary or extent. 

Clustering can be a helpful tool in your toolbox to learn more about the problem domain or for knowledge discovery.

There are many types of clustering algorithms but at this moment *SmartCore* supports only [K-means](https://en.wikipedia.org/wiki/K-means_clustering) and [DBSCAN](https://en.wikipedia.org/wiki/DBSCAN).

To fit K-means to your data use `fit` method from the [`KMeans`]({{site.api_base_url}}/cluster/kmeans/index.html) struct. Method `fit` takes a _NxM_ matrix with your data where _N_ is the number of samples and _M_ is the number of features. Another parameter of this function, _K_, is the number of clusters. If you don't know how many clusters are in your data use [Elbow Method](https://en.wikipedia.org/wiki/Elbow_method_(clustering)) to estimate it.

```rust
// Load datasets API
use smartcore::dataset::*;
// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
// K-Means
use smartcore::cluster::kmeans::{KMeans, KMeansParameters};
// Performance metrics
use smartcore::metrics::{homogeneity_score, completeness_score, v_measure_score};
// Load dataset
let digits_data = digits::load_dataset();
// Transform dataset into a NxM matrix
let x = DenseMatrix::from_array(
    digits_data.num_samples,
    digits_data.num_features,
    &digits_data.data,
);
// These are our target class labels
let true_labels = digits_data.target;
// Fit & predict
let labels = KMeans::fit(&x, KMeansParameters::default().with_k(10))
    .and_then(|kmeans| kmeans.predict(&x))
    .unwrap();
// Measure performance
println!("Homogeneity: {}", homogeneity_score(&true_labels, &labels));
println!("Completeness: {}", completeness_score(&true_labels, &labels));
println!("V Measure: {}", v_measure_score(&true_labels, &labels));
```

By default, `KMeans` terminates when it reaches 100 iterations without converging to a stable set of clusters. Pass an instance of [`KMeansParameters`]({{site.api_base_url}}/cluster/kmeans/struct.KMeansParameters.html) instead of `Default::default()` into method `fit` if you want to change value of this parameter.

The DBSCAN implementation can be found in the [dbscan]({{site.api_base_url}}/cluster/dbscan/index.html) module. To fit DBSCAN to your dataset:

```rust
// Load datasets API
use smartcore::dataset::*;
// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
// DBSCAN
use smartcore::cluster::dbscan::{DBSCANParameters, DBSCAN};
// Performance metrics
use smartcore::metrics::{completeness_score, homogeneity_score, v_measure_score};
// Load dataset
let circles = generator::make_circles(1000, 0.5, 0.05);
// Transform dataset into a NxM matrix
let x = DenseMatrix::from_array(circles.num_samples, circles.num_features, &circles.data);
// These are our target class labels
let true_labels = circles.target;
// Fit & predict
let labels = DBSCAN::fit(
    &x,
    DBSCANParameters::default()
        .with_eps(0.2)
        .with_min_samples(5),
)
.and_then(|c| c.predict(&x))
.unwrap();

// Measure performance
println!("Homogeneity: {}", homogeneity_score(&true_labels, &labels));
println!(
    "Completeness: {}",
    completeness_score(&true_labels, &labels)
);
println!("V Measure: {}", v_measure_score(&true_labels, &labels));
utils::scatterplot(
    &x,
    Some(&labels.into_iter().map(|f| f as usize).collect()),
    "test",
)
.unwrap();
```

DBSCAN is good for data which contains clusters of similar density. If you visualize results using a scatter plot you will see that each concentric circle is assigned to a separate cluster.

<figure class="image" align="center">
  <img src="{{site.baseurl}}/assets/imgs/circles.svg" alt="DBSCAN" class="img-fluid img-thumbnail">
  <figcaption>Figure 1. DBSCAN results when applied to a toy dataset.</figcaption>
</figure>

## Dimensionality Reduction

A large number of correlated variables in the feature space can dramatically impact the performance of machine learning algorithms (see [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality)). Therefore, it is often desirable to reduce the dimensionality of the feature space.

Principal component analysis (PCA) is a popular approach to dimensionality reduction from the field of linear algebra. PCA is often called "feature projection" and the algorithms used are referred to as "projection methods".

PCA is an unsupervised approach, since it involves only a set of \\(n\\) features \\(x_1, x_2, . . . , x_n\\), and no associated response \\(y_i\\). Apart from producing uncorrelated variables for use in supervised learning problems, PCA also serves as a tool for data visualization.

In PCA, the set of features \\(x_i\\) is re-expressed in terms of a set of an equal number of principal component variables. Whereas the features might be intercorrelated, the principal component variables are not. Each of the principal components found by PCA is a linear combination of the \\(n\\) features. The first principal component has the largest variance, the second component has the second largest variance, and so on.

In *SmartCore*, PCA is declared in [`pca`]({{site.api_base_url}}/decomposition/pca/index.html) module. Here is how you can calculate the first two principal components for the [Digits]({{site.api_base_url}}/dataset/digits/index.html) dataset:

```rust
use smartcore::dataset::*;
// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
// PCA
use smartcore::decomposition::pca::{PCA, PCAParameters};
// Load dataset
let digits_data = digits::load_dataset();
// Transform dataset into a NxM matrix
let x = DenseMatrix::from_array(
    digits_data.num_samples,
    digits_data.num_features,
    &digits_data.data,
);
// These are our target class labels
let labels = digits_data.target;
// Fit PCA to digits dataset
let pca = PCA::fit(&x, PCAParameters::default().with_n_components(2)).unwrap();
// Reduce dimensionality of X to 2 principal components
let x_transformed = pca.transform(&x).unwrap();
```

Once you've reduced the set of input features to the first two principal components you can visualize your data using a scatter plot, similar to <nobr>Figure 2</nobr>. 

<figure class="image" align="center">
  <img src="{{site.baseurl}}/assets/imgs/digits_pca.svg" alt="PCA" class="img-fluid img-thumbnail">
  <figcaption>Figure 2. First two principal components of the Digits dataset.</figcaption>
</figure>

Singular Value Decomposition (SVD) is another popular technique for dimensionality reduction. Use [`svd`]({{site.api_base_url}}/decomposition/svd/index.html) module to reduce dimentions for the Digits dataset:

```rust
// Load datasets API
use smartcore::dataset::*;
// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
// SVD
use smartcore::decomposition::svd::{SVDParameters, SVD};
// Load dataset
let digits_data = digits::load_dataset();
// Transform dataset into a NxM matrix
let x = DenseMatrix::from_array(
    digits_data.num_samples,
    digits_data.num_features,
    &digits_data.data,
);
// These are our target class labels
let labels = digits_data.target;
// Fit SVD to digits dataset
let svd = SVD::fit(&x, SVDParameters::default().with_n_components(2)).unwrap();
// Reduce dimensionality of X
let x_transformed = svd.transform(&x).unwrap();
```

## Matrix Factorization

Many complex matrix operations cannot be solved efficiently or with stability using the limited precision of computers. One way to solve this problem is to use matrix decomposition methods (or matrix factorization methods) that reduce a matrix into its constituent parts.

Matrix decomposition methods are at the foundation of basic operations such as solving systems of linear equations, calculating the inverse, and calculating the determinant of a matrix.

*SmartCore* supports a variety of matrix factorization methods:

{:.table .table-striped .table-bordered}
| Method | Description | Implementation |
|:-:|-|-|
| QR | \\(A=QR\\), orthonormal columns in \\(Q\\), upper triangular \\(R\\) | [`QRDecomposableMatrix`]({{site.api_base_url}}/linalg/qr/index.html) |
| LU | \\(A=LU\\), lower triangular \\(L\\), upper triangular \\(Q\\) | [`LUDecomposableMatrix`]({{site.api_base_url}}/linalg/lu/index.html) |
| EVD | \\(A=Q \Lambda Q^{-1}\\), eigenvectors in \\(Q\\), eigenvalues in \\(\Lambda\\), left eigenvectors in \\(Q^{-1}\\) | [`EVDDecomposableMatrix`]({{site.api_base_url}}/linalg/evd/index.html) |
| SVD | \\(A = U \Sigma V^T\\), columns of \\(U\\) are left-singular vectors of _A_, \\(V\\) are right-singular vectors of _A_, diagonal values in the \\(\Sigma\\) are singular values of _A_ | [`SVDDecomposableMatrix`]({{site.api_base_url}}/linalg/svd/index.html) |

Here is how you can decompose feature space of [Digits]({{site.api_base_url}}/dataset/digits/index.html) dataset into \\(U \Sigma V^T\\) using SVD:

```rust
use smartcore::dataset::*;
// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
// SVD
use smartcore::linalg::BaseMatrix;
use smartcore::linalg::svd::SVDDecomposableMatrix;
// Load dataset
let digits_data = digits::load_dataset();
// Transform dataset into a NxM matrix
let x = DenseMatrix::from_array(
    digits_data.num_samples,
    digits_data.num_features,
    &digits_data.data,
);
// Decompose matrix into U . Sigma . V^T
let svd = x.svd().unwrap();    
let u: &DenseMatrix<f32> = &svd.U; //U
let v: &DenseMatrix<f32> = &svd.V; // V
let s: &DenseMatrix<f32> = &svd.S(); // Sigma
// Print dimensions of components
println!("U is {}x{}", u.shape().0, u.shape().1);
println!("V is {}x{}", v.shape().0, v.shape().1);
println!("sigma is {}x{}", s.shape().0, s.shape().1);    
// Restore original matrix
let x_hat = u.matmul(s).matmul(&v.transpose());
for (x_i, x_hat_i) in x.iter().zip(x_hat.iter()){
    assert!((x_i - x_hat_i).abs() < 1e-3)
} 
``` 

## References
* ["An Introduction to Statistical Learning", James G., Witten D., Hastie T., Tibshirani R.](http://faculty.marshall.usc.edu/gareth-james/ISL/)
* ["The Statistical Sleuth A Course in Methods of Data Analysis", Ramsey F. L., Schafer D.W, 3rd ed.](http://www.statisticalsleuth.com/)
* ["Linear Algebra and Its Applications", Gilbert Strang, 5th ed.](https://www.academia.edu/32459792/_Strang_G_Linear_algebra_and_its_applications_4_5881001_PDF)
