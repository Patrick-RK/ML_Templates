# Machine Learning Model Templates

## Overview

This repository contains templates for different types of machine learning models, each designed with fundamental hyperparameters and tailored visualizations. It aims to provide a comprehensive starting point for understanding how different models work and how data transformations affect results.

## Structure

- **Modular:** Easily adjust hyperparameters.
- **Data-agnostic:** Works with any clean dataset, provided the target variable is defined.
- **Visual:** Outputs charts and metrics for performance insights.

## How to Use

1. **Prepare your dataset:**
   - Ensure the data is clean and preprocessed.
   - Define the target variable column for training.

2. **Select a model:**
   - Browse through the folders for various models like logistic regression, neural networks, etc.

3. **Run the model:**
   - Load your dataset into the template.
   - Set the target variable.
   - Adjust hyperparameters as needed.
   - Execute the script to generate performance metrics and visualizations.

## Model Descriptions

### 1. Logistic Regression
   - **Hyperparameters:** 
     - `penalty`: 'l1', 'l2', 'elasticnet', or 'none' – Type of regularization.
     - `solver`: 'liblinear', 'saga', etc. – Optimization algorithm.
     - `max_iter`: Number of iterations for convergence.
   - **Specific Visualizations:** 
     - *ROC Curve:* Evaluates classification performance.
     - *Precision-Recall Curve:* For imbalanced datasets.

### 2. Linear Regression
   - **Hyperparameters:** 
     - `fit_intercept`: Whether to include an intercept term.
     - `normalize`: Whether to normalize input variables.
   - **Specific Visualizations:** 
     - *Residual Plot:* Visualizes errors.
     - *Regression Line Plot:* Shows line fitting.

### 3. Decision Tree
   - **Hyperparameters:** 
     - `criterion`: 'gini' or 'entropy' – Function to measure the quality of a split.
     - `max_depth`: Maximum depth of the tree.
     - `min_samples_split`: Minimum samples required to split an internal node.
   - **Specific Visualizations:** 
     - *Tree Plot:* Visualizes decision nodes.
     - *Feature Importance Plot:* Shows significant features.

### 4. Random Forests
   - **Hyperparameters:** 
     - `n_estimators`: Number of trees in the forest.
     - `max_features`: Number of features to consider for the best split.
   - **Specific Visualizations:** 
     - *Feature Importance Plot:* Highlights key features across trees.
     - *Out-of-Bag Error Plot:* For performance validation.

### 5. Support Vector Machines (SVM)
   - **Hyperparameters:** 
     - `kernel`: 'linear', 'rbf', etc. – Type of kernel function.
     - `C`: Regularization parameter.
     - `gamma`: Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
   - **Specific Visualizations:** 
     - *Decision Boundary Plot:* Shows separation of classes.
     - *Support Vectors Plot:* Displays support vectors affecting the decision boundary.

### 6. K-Nearest Neighbour (KNN)
   - **Hyperparameters:** 
     - `n_neighbors`: Number of neighbors to use.
     - `weights`: 'uniform' or 'distance' – Weight function in prediction.
     - `metric`: 'euclidean', 'manhattan', etc. – Distance metric.
   - **Specific Visualizations:** 
     - *K vs Accuracy Plot:* Evaluates varying K.
     - *Decision Boundary Plot:* Shows neighborhood influence on classification.

### 7. Neural Networks
   - **Hyperparameters:** 
     - `hidden_layer_sizes`: Number of neurons in hidden layers.
     - `activation`: 'relu', 'tanh', etc. – Activation function.
     - `learning_rate_init`: Initial learning rate.
   - **Specific Visualizations:** 
     - *Loss Curve:* Tracks loss during training.
     - *Confusion Matrix:* Evaluates classification results.

### 8. Principal Component Analysis (PCA)
   - **Hyperparameters:** 
     - `n_components`: Number of components to keep.
     - `svd_solver`: 'auto', 'full', etc. – Algorithm for decomposition.
   - **Specific Visualizations:** 
     - *Explained Variance Plot:* Shows variance explained by components.
     - *Biplot:* Combines PCA components with feature projections.

### 9. K-Means Clustering
   - **Hyperparameters:** 
     - `n_clusters`: Number of clusters.
     - `init`: Initialization method – 'k-means++', 'random', etc.
     - `max_iter`: Maximum number of iterations.
   - **Specific Visualizations:** 
     - *Cluster Centers Plot:* Displays centroids of clusters.
     - *Elbow Plot:* Determines the optimal number of clusters.

## Common Visualizations Across All Models

These visualizations are commonly used across most machine learning models:

- **Confusion Matrix:** Useful for evaluating classification performance, showing true positives, false positives, true negatives, and false negatives.
- **Accuracy Score:** Indicates the overall performance of classification models.
- **ROC Curve & AUC Score:** Used to evaluate the trade-off between sensitivity and specificity in classification models.
- **Learning Curve:** Shows model performance as the size of the training dataset increases, helping to diagnose overfitting or underfitting.
- **Cross-Validation Score Plot:** Demonstrates performance stability across different folds, applicable to all model types.
- **Feature Importance Plot:** For models that support it (e.g., decision trees, random forests, etc.), shows the significance of features in predictions.
- **Residual Distribution Plot:** Useful in regression models to understand the distribution of errors.
- **Predicted vs. Actual Plot:** Visualizes how well the model's predictions align with actual values (regression).

## Contributing

If you have suggestions for improving the models or visualizations, or want to add new models, please open a pull request.
