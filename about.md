---
layout: manual
title: About
---

# About us

SmartCore is developed and maintained by Smartcore developers. Our goal is to build an open library that has accurate, numerically stable, and well-documented implementations of the most well-known and widely used machine learning algorithms. 

## Contributors

<div class="authors-container mt-3">
    {% for contributor in site.data.contributors %}
        <div class="col-lg-2 text-center text-capitalize">
            <a href="https://github.com/{{contributor.id}}">{% avatar user=contributor.id size=60 %}</a>
            <p>{{contributor.name}}</p>
        </div>
    {% endfor %}
</div>
## Release Notes

### Version 0.1.0

This is our first realease, enjoy! In this version you'll find:
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