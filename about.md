---
layout: manual
title: About
---

# About us

SmartCore is developed and maintained by Smartcore developers. Our goal is to build an open library that has accurate, numerically stable, and well-documented implementations of the most well-known and widely used machine learning algorithms. 

## Contributors

<div class="row authors-container">    
    {% for contributor in site.data.contributors %}
        <div class="col-lg-2 text-center text-capitalize">
            <a href="https://github.com/{{contributor.id}}">{% avatar user=contributor.id size=60 %}</a>
            <p>{{contributor.name}}</p>
        </div>
    {% endfor %}    
</div>
## Release Notes

### Version 0.2.0

- DBSCAN
- Epsilon-SVR, SVC
- Ridge, Lasso, ElasticNet
- Bernoulli, Gaussian, Categorical and Multinomial Naive Bayes
- K-fold Cross Validation
- Singular value decomposition
- New api module
- Integration with Clippy
- smartcore::error:FailedError is now non-exhaustive 
- ndarray upgraded to 0.14
- Cholesky decomposition
- API changed in: K-Means, PCA, Random Forest, Linear and Logistic Regression, KNN, Decision Tree

### Version 0.1.0

This is our first release, enjoy! In this version you'll find:
- KNN + distance metrics (Euclidian, Minkowski, Manhattan, Hamming, Mahalanobis)
- Linear Regression (OLS)
- Logistic Regression
- Random Forest Classifier
- Decision Tree Classifier
- PCA
- K-Means
- Integrated with ndarray
- Abstract linear algebra methods
- RandomForest Regressor
- Decision Tree Regressor
- Serde integration
- Integrated with nalgebra
- LU, QR, SVD, EVD
- Evaluation Metrics

Please let us know if you found a problem. The best way to report it is to [open an issue](https://github.com/smartcorelib/smartcore/issues) on GitHub.
