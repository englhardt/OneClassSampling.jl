# One-Class Sampling
_A Julia package for Sampling Strategies for One-Class Classification._

This package implements sampling strategies for one-class classification.
For more information about this research project, see the [project](https://www.ipd.kit.edu/ocs/) website, and the companion paper.

> Adrian Englhardt, Holger Trittenbach, Daniel Kottke, Bernhard Sick, Klemens Böhm, "Efficient SVDD Sampling with Approximation Guarantees for the Decision Boundary", 11 June 2020

## Installation
This package requires at least Julia 1.3.
This package is not registered yet. Please use the following command to install the package.
```Julia
using Pkg
Pkg.add("https://github.com/englhardt/OneClassSampling.jl.git")
```

## Overview
[One-class classifiers](https://en.wikipedia.org/wiki/One-class_classification) learn to identify if objects belong to a specific class, often used for outlier detection.
One popular unsupervised one-class classifier is Support Vector Data Description (SVDD)[1].
SVDD fits a tight hypersphere around the majority of the observations, the inliers, and exclude some, the outliers.
One downside of SVDD is that it does not scale well with data size.
Therefore, sampling strategies are essential to reduce the data size before training.

## Example

The following snippet runs our novel sampling strategy RAPID on a small data set.
Here, we pass RAPID the parameters for the threshold strategy (0% outlier percentage) and the heuristic to choose gamma for the KDE (Modified Mean Criterion [2]):
```Julia
data, labels = randn(5, 1000), fill(:inlier, 1000)
sampler = RAPID(OutlierPercentageThresholdStrategy(0.0), :mod_mean_crit)
sample_mask = sample(sampler, data, labels)
data_sampled, labels_sampled = data[:, sample_mask], labels[sample_mask]
```

## Implemented Sampling Strategies

* RSSVDD [3]
* CSSVDD [4]
* KMSVDD [5]
* BPS [6]
* DAEDS [7]
* DBSRSVDD [8]
* FBPE [9]
* HSR [10]
* HSC [11]
* IESRSVDD [12]
* KFNCBD [13]
* NDPSR [14]
* RAPID (our novel strategy)

## Density parameters

This package uses the Gaussian Kernel and one can select the γ parameter in the following ways (wrapped from the [SVDD.jl](https://github.com/englhardt/SVDD.jl) package)
* `:scott` Scotts's rule [15]
* `:mean_crit` Mean Criterion [16]
* `:mod_mean_crit` Modified Mean Criterion [2]
* manual mode: simply pass any `float`

Available threshold strategies:

* `FixedThresholdStrategy(x)`: manually pass a float `x`
* `MaxDensityThresholdStrategy(eps)` calculates the threshold by multiplying the maximum density with `eps` (`eps` must be in `[0, 1]`).
* `OutlierPercentageThresholdStrategy(eps)` calculates the threshold as the eps-th quantile of the density (`eps` must be in `[0, 1]`). When one passes `eps=nothing` the threshold calculation takes the ground truth outlier percentage.
* `GroundTruthThresholdStrategy()` calculates the threshold according to the ground truth.

## Authors
We welcome contributions and bug reports.

This package is developed and maintained by [Adrian Englhardt](https://github.com/englhardt).

## References
[1] D. Tax and R. Duin, "Support vector data description," Machine Learning, 2004.<br>
[2] Y. Liao, D. Kakde, A. Chaudhuri, H. Jiang, C. Sadek, and S. Kong, "A new bandwidth selection criterion for using SVDD to analyze hyperspectral data," Algorithms and Technologies for Multispectral, Hyperspectral, and Ultraspectral Imagery. SPIE, 2018.<br>
[3] A. Chaudhuri, D. Kakde, M. Jahja, W. Xiao, S. Kong, H. Jiang, S. Percdriy, "Sampling method for fast training of support vector data description," RAMS. IEEE, 2018.<br>
[4] C. S. Chu, I. W. Tsang, and J. T. Kwok, "Scaling up support vector data description by using core-sets," IJCNN. IEEE, 2004.<br>
[5] P. J. Kim, H. J. Chang, D. S. Song, and J. Y. Choi, "Fast support vector data description using k-means clustering," ISNN. Springer, 2007<br>
[6] Y. Li, "Selecting training points for one-class support vector machines," Pattern Recognition Letters, 2011.<br>
[7] C. Hu, B. Zhou, and J. Hu, "Fast support vector data description training using edge detection on large datasets," IJCNN. IEEE, 2014.<br>
[8] Z. Li, L. Wang, Y. Yang, X. Du, and H. Song, "Health evaluation of mvb based on svdd and sample reduction," IEEE Access, 2019.<br>
[9] S. Alam, S. K. Sonbhadra, S. Agarwal, P. Nagabhushan, and M. Tanveer, "Sample reduction using farthest boundary point estimation (fbpe) for support vector data description (svdd)," Pattern Recognition Letters, 2020.<br>
[10] W. Sun, J. Qu, Y. Chen, Y. Di, and F. Gao, "Heuristic sample reduction method for support vector data description," Turkish Journal of Electrical Engineering & Computer Sciences, 2016.<br>
[11] H. Qu, J. Zhao, J. Zhao, and D. Jiang, "Towards support vector data description based on heuristic sample condensed rule," CCDC. IEEE, 2019.<br>
[12] D. Li, Z. Wang, C. Cao, and Y. Liu, "Information entropy based sample reduction for support vector data description," Applied Soft Computing, 2018.<br>
[13] Y. Xiao, B. Liu, Z. Hao, and L. Cao, "A k-farthest-neighbor-based approach for support vector data description," Applied intelligence, 2014.<br>
[14] F. Zhu, N. Ye, W. Yu, S. Xu, and G. Li, "Boundary detection and sample reduction for one-class support vector machines," Neurocomputing, 2014.<br>
[15] D. W. Scott, "Multivariate density estimation: theory, practice, and visualization," John Wiley & Sons, 2015.<br>
[16] A. Chaudhuri, D. Kakde, C. Sadek, L. Gonzalez, and S. Kong, "The mean and median criteria for kernel bandwidth selection for support vector data description," ICDM Workshop, 2017.<br>
