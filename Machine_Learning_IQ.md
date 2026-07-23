# Interview Question Bank

---

# Bias-Variance Tradeoff

## Primary Question: What's the trade-off between bias and variance?

### Answer
If our model is too simple and has very few parameters then it may have high bias and low variance. On the other hand if our model has large number of parameters then it's going to have high variance and low bias. So we need to find the right/good balance without overfitting and underfitting the data.

The bias-variance tradeoff is a fundamental concept in machine learning that describes the tradeoff between two sources of error that affect model performance.

**Bias:**
- Error due to wrong assumptions in the learning algorithm.
- High bias means underfitting and model is too simple to capture patterns.
- Example: Linear model trying to fit highly non-linear data.

**Variance:**
- Error due to model being too sensitive to small fluctuations in training data.
- High variance means overfitting and model performs well on training data but poorly on unseen data.
- Example: Deep decision tree memorizing training data.

**Tradeoff:**
- Decreasing bias usually increases variance and vice versa.
- Our goal is to find a balance that minimizes total error.

### Formula
Total Error = Bias² + Variance + Irreducible Error

### Quick Explanation (AI Generated)
Bias is error from overly simple assumptions; variance is error from being overly sensitive to the training data. Good models balance the two to minimize total generalization error.

### Interview Tip (AI Generated)
Be ready to draw the classic bias-variance curve (U-shaped total error) and connect it to overfitting/underfitting and model complexity.

---

# Overfitting and Underfitting

## Primary Question: Explain over- and under-fitting and how to combat them?

### Answer
ML/DL models essentially learn a relationship between its given inputs(called training features) and objective outputs(called labels). Regardless of the quality of the learned relation(function), its performance on a test set(a collection of data different from the training input) is subject to investigation.

Most ML/DL models have trainable parameters which will be learned to build that input-output relationship. Based on the number of parameters each model has, they can be sorted into more flexible(more parameters) to less flexible(less parameters).

The problem of Underfitting arises when the flexibility of a model(its number of parameters) is not adequate to capture the underlying pattern in a training dataset. Overfitting, on the other hand, arises when the model is too flexible to the underlying pattern. In the later case it is said that the model has "memorized" the training data.

**Overfitting**: It occurs when a model not only learns the true patterns in the training data but also memorizes the noise or random fluctuations. This results in high accuracy on training data but poor performance on unseen/test data.

Ways to Avoid Overfitting:
- Early Stopping: Stop training when validation accuracy stops improving, even if training accuracy is still increasing.
- Regularization: Apply techniques like L1 (Lasso) or L2 (Ridge) regularization which add penalties to large weights to reduce model complexity.
- Cross-Validation: Use k-fold cross-validation to ensure the model generalizes well.
- Dropout (for Neural Networks): Randomly drop neurons during training to prevent over-reliance on specific nodes.
- Simpler Models: Avoid overly complex models when simpler ones can explain the data well.

**Underfitting**: It occurs when a model is too simple to capture the underlying patterns in the data. This leads to poor accuracy on both training and test data.

Ways to Avoid Underfitting:
- Use a More Complex Model: Choose models with higher complexity to learn pattern like decision trees, neural networks, etc.
- Add Relevant Features: Include meaningful features that better represent the data.
- Reduce Regularization: Too much regularization can restrict the model's ability to learn.
- Train Longer: Allow the model more epochs or iterations to properly learn patterns.

### Example
An example of underfitting is estimating a second order polynomial(quadratic function) with a first order polynomial(a simple line). Similarly, estimating a line with a 10th order polynomial would be an example of overfitting.

---

# Regularization

## Primary Question: What is regularization, why do we use it, and give some examples of common methods?

### Answer
A technique that discourages learning a more complex or flexible model, so as to avoid the risk of overfitting.

Regularization is a technique used to avoid overfitting by trying to make the model more simple. It is a technique used to reduce model complexity and prevent overfitting. It works by adding a penalty term to the loss function to discourage the model from assigning too much importance (large weights) to specific features. This helps the model generalize better on unseen data.

Examples/Ways to Apply Regularization:
- Ridge (L2 norm) Regularization: Adds the squared value of weights as a penalty which reduces large weights but doesn't eliminate them.
- Lasso (L1 norm) Regularization: Adds the absolute value of weights as a penalty which can shrink some weights to zero and perform feature selection.
- Elastic Net: Combines both L1 and L2 penalties to balance feature selection and weight reduction.
- Dropout (for Neural Networks): Randomly drops neurons during training to avoid over-reliance on specific nodes.

The obvious disadvantage of ridge regression, is model interpretability. It will shrink the coefficients for least important predictors, very close to zero. But it will never make them exactly zero. In other words, the final model will include all predictors. However, in the case of the lasso, the L1 penalty has the effect of forcing some of the coefficient estimates to be exactly equal to zero when the tuning parameter λ is sufficiently large. Therefore, the lasso method also performs variable selection and is said to yield sparse models.

## Similar Question: What are L1 and L2 regularization? What are the differences between the two?

### Answer
**Lasso Regularization (L1):** Lasso adds a penalty equal to the absolute value of the model's weights to the loss function. It can shrink some weights to exactly zero, performing feature selection.

**Ridge Regularization (L2):** It adds a penalty equal to the square of the model's weights to the loss function. It reduces large weights but does not set them to zero, helping generalization.

Key Differences:
- Lasso (L1): Can set weights to zero → feature selection. Use it when we have many irrelevant features.
- Ridge (L2): Reduces weights but keeps all features → no feature elimination. Use when all features are useful but want to avoid overfitting.

**Elastic Net Regularization:** Elastic Net combines both L1 (Lasso) and L2 (Ridge) penalties, balancing feature selection and weight reduction. It is especially useful when features are correlated, as it avoids Lasso's limitation of picking only one feature from a group.

### Formula
Lasso Loss = MSE + λ∑|wᵢ|

Ridge Loss = MSE + λ∑wᵢ²

Where: MSE = Mean Squared Error, wᵢ = model weights, λ = regularization strength

---

# Gradient Descent (Types & Variants)

## Primary Question: What is gradient descent?

### Answer
Gradient descent is an optimization algorithm used to find the values of parameters (coefficients) of a function (f) that minimizes a cost function (cost).

Gradient descent is best used when the parameters cannot be calculated analytically (e.g. using linear algebra) and must be searched for by an optimization algorithm.

Gradient descent is a generic optimization algorithm capable of finding optimal solutions to a wide range of problems. The general idea of gradient descent is to tweak parameters iteratively in order to minimize a cost function.

Gradient Descent is an optimization algorithm used to minimize the loss function by iteratively updating model parameters in the direction opposite to the gradient of the loss with respect to those parameters.

### Formula
Update Rule: θ = θ − α · ∇J(θ)

Where: θ = model parameters (weights), α = learning rate, ∇J(θ) = gradient of the loss function

## Similar Question: Explain briefly batch gradient descent, stochastic gradient descent, and mini-batch gradient descent, and what are the pros and cons for each of them?

### Answer
**Batch Gradient Descent:**
In Batch Gradient descent the whole training data is used to minimize the loss function by taking a step toward the nearest minimum by calculating the gradient (the direction of descent).

Pros: Since the whole data set is used to calculate the gradient it will be stable and reach the minimum of the cost function without bouncing (if the learning rate is chosen correctly).

Cons: Since batch gradient descent uses all the training set to compute the gradient at every step, it will be very slow especially if the size of the training data is large. Uses the entire training dataset to compute the gradient at each step. Stable convergence but slow and memory-heavy for large datasets.

**Stochastic Gradient Descent:**
Stochastic Gradient Descent picks up a random instance in the training data set at every step and computes the gradient based only on that single instance. Updates parameters using one training example at a time.

Pros:
1. It makes the training much faster as it only works on one instance at a time.
2. It becomes easier to train large datasets.
3. Much faster per step and can escape shallow local minima.

Cons: Due to the stochastic (random) nature of this algorithm, this algorithm is much less regular than the batch gradient descent. Instead of gently decreasing until it reaches the minimum, the cost function will bounce up and down, decreasing only on average. Over time it will end up very close to the minimum, but once it gets there it will continue to bounce around, not settling down there. So once the algorithm stops the final parameters are good but not optimal. For this reason, it is important to use a training schedule to overcome this randomness. The path to convergence is noisy.

**Mini-batch Gradient:**
At each step instead of computing the gradients on the whole data set as in the Batch Gradient Descent or using one random instance as in the Stochastic Gradient Descent, this algorithm computes the gradients on small random sets of instances called mini-batches. Uses a small batch of samples (e.g., 32, 64, 128) per update.

Pros:
1. The algorithm's progress space is less erratic than with Stochastic Gradient Descent, especially with large mini-batches.
2. You can get a performance boost from hardware optimization of matrix operations, especially when using GPUs.
3. Balances the stability of batch GD with the speed of SGD — the most commonly used approach in practice.

Cons: It might be difficult to escape from local minima.

### Diagram
![alt text](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Figures/gradient%20descent%20vs%20batch%20gradient%20descent.png)

## Similar Question: What is the importance of batch in machine learning and explain some batch dependent gradient descent algorithms?

### Answer
In the memory, the dataset can load either completely at once or in the form of a set. If we have a huge size of the dataset, then loading the whole data into memory will reduce the training speed, hence batch term is introduced.

In summary, a batch is important in two ways: (1) Efficient memory consumption. (2) Improve training speed.

There are 3 types of gradient descent algorithms based on batch size: (1) Stochastic gradient descent (2) Batch gradient descent (3) Mini Batch gradient descent

If the whole data is in a single batch, it is called batch gradient descent. If the single data points are equal to one batch i.e. number of batches = number of data instances, it is called stochastic gradient descent. If the number of batches is less than the number of data points or greater than 1, it is known as mini-batch gradient descent.

### Example
Image data contains 1,00,000 images, we can load this into 3125 batches where 1 batch = 32 images. So instead of loading the whole 1,00,000 images in memory, we can load 32 images 3125 times which requires less memory.

---

# Cost Function vs Loss Function

## Primary Question: What are Loss Functions and Cost Functions? Explain the key Difference Between them.

### Answer
The loss function is the measure of the performance of the model on a single training example, whereas the cost function is the average loss function over all training examples or across the batch in the case of mini-batch gradient descent.

Some examples of loss functions are Mean Squared Error, Binary Cross Entropy, etc.

Whereas, the cost function is the average of the above loss functions over training examples.

Cost function is a scalar function which quantifies the error factor of the Neural Network. Lower the cost function better the Neural network.

### Example
Eg: MNIST Data set to classify the image, input image is digit 2 and the Neural network wrongly predicts it to be 3.

### Quick Explanation (AI Generated)
Different Loss Functions in Machine Learning (from notes):
- Mean Squared Error (MSE): Used in regression, penalizes larger errors more heavily.
- Mean Absolute Error (MAE): Used in regression, less sensitive to outliers than MSE.
- Huber Loss: Combines MSE and MAE, less sensitive to outliers than MSE.
- Cross-Entropy Loss (Log Loss): Used in classification, measures difference between predicted probability distribution and actual labels.
- Hinge Loss: Used for classification with SVMs, encourages maximum margin between classes.
- KL Divergence: Measures how one probability distribution differs from another.
- Exponential Loss: Used in boosting methods like AdaBoost; penalizes misclassified points more strongly.

### Formula
MSE = (1/n)∑(yᵢ − ŷᵢ)²

MAE = (1/n)∑|yᵢ − ŷᵢ|

CE = −(1/n)∑[yᵢlog(ŷᵢ) + (1−yᵢ)log(1−ŷᵢ)]

---

# Curse of Dimensionality

## Primary Question: How do you combat the curse of dimensionality?

### Answer
- Feature Selection(manual or via statistical methods)
- Principal Component Analysis (PCA)
- Multidimensional Scaling
- Locally linear embedding

Curse of dimensionality represents the situation when the amount of data is too few to be represented in a high-dimensional space, as it will be highly scattered in that high-dimensional space and it becomes more probable that we overfit this data. If we increase the number of features, we are implicitly increasing model complexity and if we increase model complexity we need more data.

Possible solutions are: Remove irrelevant features not discriminating classes correlated or features not resulting in much improvement, we can use:
- Feature selection(select the most important ones).
- Feature extraction(transform current feature dimensionality into a lower dimension preserving the most possible amount of information like PCA).

The curse of dimensionality refers to the various problems that arise when working with data that has a very large number of features (high dimensions). As dimensions increase, the volume of the space grows exponentially, and data points become sparse and spread far apart.
- Distance metrics lose meaning: In high dimensions, the difference between the nearest and farthest neighbor's distance shrinks, making distance-based algorithms like KNN and K-Means less effective.
- Increased computational cost: More features mean more computation for training and distance calculations.
- Overfitting risk: With many features but limited samples, models can easily fit noise instead of true patterns.
- Data sparsity: Points become isolated from each other, making it harder to find meaningful clusters or neighbors.

### Diagram
![Curse of dim'](https://user-images.githubusercontent.com/72076328/188653089-8999ea59-9511-4d52-baff-15a652e117a9.png)

---

# Principal Component Analysis (PCA)

## Primary Question: Explain Principal Component Analysis (PCA)?

### Answer
Principal Component Analysis (PCA) is a dimensionality reduction technique used in machine learning to reduce the number of features in a dataset while retaining as much information as possible. It works by identifying the directions (principal components) in which the data varies the most, and projecting the data onto a lower-dimensional subspace along these directions.

Principal Component Analysis (PCA) is a dimensionality reduction technique used in machine learning and statistics. It transforms a high-dimensional dataset into a lower-dimensional space while preserving as much variance (information) as possible.

- Reduces computational complexity by lowering dimensions.
- Removes redundant or correlated features.
- Helps in visualizing high-dimensional data.
- Principal components may lack interpretability.
- Assumes linear relationships between features.
- Sensitive to scaling of features and standardization is necessary.

Working of PCA:
1. Standardize the data to have mean 0 and variance 1.
2. Compute the covariance matrix of the features.
3. Calculate the eigenvalues and eigenvectors of the covariance matrix.
4. Eigenvectors define the directions (principal components).
5. Eigenvalues indicate the amount of variance in each direction.
6. Select top k principal components with the highest eigenvalues.
7. Transform the original data onto the new k-dimensional space.

PCA focuses on directions with highest variance, as variance represents information content. By projecting data onto these directions: Information is preserved while reducing dimensionality. Less important features with low variance are discarded, simplifying the data.

### Example
Dataset with features like height, weight, age and income. PCA can reduce it to 2 or 3 principal components that capture most of the variance for visualization or modeling.

## Similar Question: In what cases would you use vanilla PCA, Incremental PCA, Randomized PCA, or Kernel PCA?

### Answer
Regular PCA is the default, but it works only if the dataset fits in memory. Incremental PCA is useful for large datasets that don't fit in memory, but it is slower than regular PCA, so if the dataset fits in memory you should prefer regular PCA. Incremental PCA is also useful for online tasks when you need to apply PCA on the fly, every time a new instance arrives. Randomized PCA is useful when you want to considerably reduce dimensionality and the dataset fits in memory; in this case, it is much faster than regular PCA. Finally, Kernel PCA is useful for nonlinear datasets.

---

# t-SNE

## Primary Question: What is t-SNE?

### Answer
t-Distributed Stochastic Neighbor Embedding (t-SNE) is an unsupervised, non-linear technique primarily used for data exploration and visualizing high-dimensional data. In simpler terms, t-SNE gives you a feel or intuition of how the data is arranged in a high-dimensional space.

---

# UMAP

## Primary Question: What is UMAP?

### Answer
UMAP (Uniform Manifold Approximation and Projection) is a novel manifold learning technique for dimension reduction. UMAP is constructed from a theoretical framework based in Riemannian geometry and algebraic topology. The result is a practical scalable algorithm that applies to real world data.

---

# Dimensionality Reduction Technique Comparisons

## Primary Question: What is the difference between LDA and PCA for dimensionality reduction?

### Answer
Both LDA and PCA are linear transformation techniques: LDA is a supervised whereas PCA is unsupervised – PCA ignores class labels.

We can picture PCA as a technique that finds the directions of maximal variance. In contrast to PCA, LDA attempts to find a feature subspace that maximizes class separability.

## Similar Question: What is the difference between t-SNE and PCA for dimensionality reduction?

### Answer
The first thing to note is that PCA was developed in 1933 while t-SNE was developed in 2008. A lot has changed in the world of data science since 1933 mainly in the realm of compute and size of data. Second, PCA is a linear dimension reduction technique that seeks to maximize variance and preserves large pairwise distances. In other words, things that are different end up far apart. This can lead to poor visualization especially when dealing with non-linear manifold structures. Think of a manifold structure as any geometric shape like: cylinder, ball, curve, etc.

t-SNE differs from PCA by preserving only small pairwise distances or local similarities whereas PCA is concerned with preserving large pairwise distances to maximize variance.

## Similar Question: What is the difference between t-SNE and UMAP for dimensionality reduction?

### Answer
The biggest difference between the output of UMAP when compared with t-SNE is this balance between local and global structure - UMAP is often better at preserving global structure in the final projection. This means that the inter-cluster relations are potentially more meaningful than in t-SNE. However, it's important to note that, because UMAP and t-SNE both necessarily warp the high-dimensional shape of the data when projecting to lower dimensions, any given axis or distance in lower dimensions still isn't directly interpretable in the way of techniques such as PCA.

---

# Dimensionality Reduction (General)

## Primary Question: What is Dimensionality Reduction in Machine Learning?

### Answer
Dimensionality reduction is the process of reducing the number of features (variables) in a dataset while retaining most of the important information.

- It helps in simplifying models, improving performance, reducing overfitting and speeding up computation.
- Reduces computational cost for high-dimensional datasets.
- Helps visualize data in 2D or 3D space.
- Reduces overfitting by removing irrelevant or noisy features.

### Example
A dataset has 100 features. Using PCA, it can be reduced to 10 principal components that capture 95% of the variance.

---

# Evaluating Dimensionality Reduction Algorithms

## Primary Question: How can you evaluate the performance of a dimensionality reduction algorithm on your dataset?

### Answer
Intuitively, a dimensionality reduction algorithm performs well if it eliminates a lot of dimensions from the dataset without losing too much information. One way to measure this is to apply the reverse transformation and measure the reconstruction error. However, not all dimensionality reduction algorithms provide a reverse transformation.

Alternatively, if you are using dimensionality reduction as a preprocessing step before another Machine Learning algorithm (e.g., a Random Forest classifier), then you can simply measure the performance of that second algorithm; if dimensionality reduction did not lose too much information, then the algorithm should perform just as well as when using the original dataset.

---

# ReLU vs Sigmoid Activation

## Primary Question: Why is ReLU better and more often used than Sigmoid in Neural Networks?

### Answer
- Computation Efficiency: As ReLU is a simple threshold the forward and backward path will be faster.
- Reduced Likelihood of Vanishing Gradient: Gradient of ReLU is 1 for positive values and 0 for negative values while Sigmoid activation saturates (gradients close to 0) quickly with slightly higher or lower inputs leading to vanishing gradients.
- Sparsity: Sparsity happens when the input of ReLU is negative. This means fewer neurons are firing (sparse activation) and the network is lighter.

---

# Activation Functions (List)

## Primary Question: List different activation neurons or functions.

### Answer
- Linear Neuron
- Binary Threshold Neuron
- Stochastic Binary Neuron
- Sigmoid Neuron
- Tanh function
- Rectified Linear Unit (ReLU)

---

# Machine Learning vs AI vs Data Science

## Primary Question: What do you understand by Machine Learning (ML) and how does it differ from artificial intelligence (AI) and Data Science?

### Answer
Machine Learning (ML) is a subset of Artificial Intelligence that enables computers to learn patterns from data and make predictions or decisions without being explicitly programmed.

**Artificial Intelligence (AI)**
- Artificial Intelligence is the broader field of creating machines that can mimic human intelligence.
- It focuses on tasks such as reasoning, problem-solving, planning, learning, and decision-making.
- AI includes multiple approaches such as Machine Learning, Deep Learning, rule-based systems, robotics, and expert systems.
- Goal: Build intelligent systems capable of performing tasks that normally require human intelligence.

**Machine Learning (ML)**
- Machine Learning (ML) uses algorithms to identify patterns and make predictions or decisions.
- The model improves its performance through experience instead of being explicitly programmed for every task.
- Goal: Develop models that can learn from historical data and generalize to unseen data.

**Data Science**
- Data Science is an interdisciplinary field that extracts meaningful insights from structured and unstructured data.
- It combines statistics, programming, data visualization, domain knowledge, and Machine Learning.
- Data Science involves the complete data lifecycle, including data collection, cleaning, analysis, visualization, and model building.
- Goal: Transform raw data into actionable insights for business decision-making.

---

# Model Evaluation Techniques

## Primary Question: What are different Model Evaluation Techniques in Machine Learning?

### Answer
Model evaluation techniques are used to assess how well a machine learning model performs on unseen data. Choosing the right technique depends on the type of problem like classification, regression, etc and type of dataset we have.

- Train-Test Split: Divide data into training and testing sets like 70:30 or 80:20 to evaluate model performance on unseen data. Here 70% data will be used for training and 30% will be used to test accuracy of model.
- Cross-Validation: Split data into k folds, train on k-1 folds, validate on the remaining fold and average the results to reduce bias.
- Confusion Matrix (for Classification): Counts True Positives, True Negatives, False Positives and False Negatives.
- Accuracy: Proportion of correct predictions over total predictions.
- Precision: Here correct positive predictions are divided by Total predicted positives.
- Recall (Sensitivity): Here correct positive predictions are divided by Total actual positives.
- F1-Score: Harmonic mean of precision and recall. It balances precision and recall.
- ROC Curve & AUC: Measures model's ability to distinguish between classes. Here AUC is area under the ROC curve.

---

# Confusion Matrix

## Primary Question: Explain Confusion Matrix.

### Answer
A confusion matrix is a table used to evaluate the performance of a classification model. It compares the predicted labels with the actual labels, telling how well the model is performing and what types of errors it makes.

Here:
- True Positives (TP): Correctly predicted positive cases.
- True Negatives (TN): Correctly predicted negative cases.
- False Positives (FP): Negative cases incorrectly predicted as positive.
- False Negatives (FN): Positive cases incorrectly predicted as negative.

It is used in metrics like Accuracy, Precision, Recall and F1-Score.

### Diagram
predicted_condition_2_ Confusion Matrix (as described in source notes)

---

# Precision, Recall, and F1-Score

## Primary Question: Define Precision, recall, and F1 and discuss the trade-off between them?

### Answer
Precision and recall are two classification evaluation metrics that are used beyond accuracy.

**Precision:** It is the ratio between the true positives(TP) and all the positive examples (TP+FP) predicted by the model. In other words, precision measures how many of the predicted positive examples are actually true positives. It is a measure of the model's ability to avoid false positives and make accurate positive predictions. Precision (also called positive predictive value) is the fraction of relevant instances among the retrieved instances.

**Recall:** It calculates the ratio of true positives (TP) and the total number of examples (TP+FN) that actually fall in the positive class. Recall measures how many of the actual positive examples are correctly identified by the model. It is a measure of the model's ability to avoid false negatives and identify all positive examples correctly. Recall (also known as sensitivity) is the fraction of relevant instances that have been retrieved over the total amount of relevant instances.

**F1-score:** It is the weighted average of precision and recall. It considers both false positive and false negative into account. It is used to measure the model's performance.

**Key Difference:**
- Precision is about being exact (avoiding false positives).
- Recall is about being comprehensive (avoiding false negatives).
- F1-Score (Balance of Both): Used when both precision and recall matter.

### Example
In spam detection, high precision means most emails marked as spam are truly spam. In disease detection, high recall means most sick patients are correctly identified.

### Formula
Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1-Score = 2 × (precision × recall) / (precision + recall)

---

# ROC / AUC-ROC Curve

## Primary Question: Explain how a ROC curve works.

### Answer
The ROC curve is a graphical representation of the contrast between true positive rates and the false positive rate at various thresholds. It's often used as a proxy for the trade-off between the sensitivity of the model (true positives) vs the fall-out or the probability it will trigger a false alarm (false positives).

ROC curve, Receiver Operating Characteristic curve, is a graphical representation of the model's performance where we plot the True Positive Rate (TPR) against the False Positive Rate (FPR) for different threshold values, for hard classification, between 0 to 1 based on model output.

This ROC curve is mainly used to compare two or more models. It is easy to see that a reasonable model will always give FPR less (since it's an error) than TPR so, the curve hugs the upper left corner of the square box 0 to 1 on the TPR axis and 0 to 1 on the FPR axis.

The more the AUC(area under the curve) for a model's ROC curve, the better the model in terms of prediction accuracy in terms of TPR and FPR.

Here are some benefits of using the ROC Curve:
- Can help prioritize either true positives or true negatives depending on your case study (Helps you visually choose the best hyperparameters for your case)
- Can be very insightful when we have unbalanced datasets
- Can be used to compare different ML models by calculating the area under the ROC curve (AUC)

AUC (Area Under the Curve): AUC is the area under the ROC curve. It represents the probability that a randomly chosen positive instance is ranked higher than a randomly chosen negative instance.
- AUC = 1 → Perfect classifier
- AUC = 0.5 → Random guessing
- AUC < 0.5 → Worse than random

ROC shows performance across thresholds. AUC summarizes overall model performance into a single number.

### Example
If a medical test has an AUC of 0.90, it means there's a 90% chance that the model will rank a randomly chosen diseased patient higher than a healthy one.

### Diagram
![Alt_text](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Figures/Roc_curve.svg.png)

### Formula
TPR (Recall) = TP / (TP + FN)

FPR = FP / (FP + TN)

---

# Type I vs Type II Error

## Primary Question: What's the difference between Type I and Type II error?

### Answer
Type I error is a false positive, while Type II error is a false negative. Briefly stated, Type I error means claiming something has happened when it hasn't, while Type II error means that you claim nothing is happening when in fact something is. A clever way to think about this is to think of Type I error as telling a man he is pregnant, while Type II error means you tell a pregnant woman she isn't carrying a baby.

In hypothesis testing and classification, errors occur when a model's prediction doesn't match reality. These map directly onto the confusion matrix.
- Type I Error (False Positive): Rejecting a true null hypothesis — i.e., predicting positive when the actual class is negative. Example: A medical test flags a healthy patient as having a disease.
- Type II Error (False Negative): Failing to reject a false null hypothesis — i.e., predicting negative when the actual class is positive. Example: A medical test misses an actual disease case, marking a sick patient as healthy.

---

# Is Accuracy Always a Good Metric

## Primary Question: Is accuracy always a good metric for classification performance?

### Answer
No, accuracy can be misleading, especially with imbalanced datasets. In such cases:
- Precision and Recall provide better insight into model performance.
- F1-score combines precision and recall as their harmonic mean, giving a balanced measure of model effectiveness, especially when the classes are imbalanced.

---

# Cross-Validation

## Primary Question: What is Cross-Validation?

### Answer
Cross-validation is a technique used to assess the performance of a learning model in several subsamples of training data. In general, we split the data into train and test sets where we use the training data to train our model and the test data to evaluate the performance of the model on unseen data and validation set for choosing the best hyperparameters. Now, a random split in most cases(for large datasets) is fine. However, for smaller datasets, it is susceptible to loss of important information present in the data in which it was not trained. Hence, cross-validation though computationally expensive combats this issue.

Cross-validation is a model evaluation technique used to test how well a machine learning model generalizes to unseen data. Instead of training and testing on a single split, the dataset is divided into multiple subsets (called folds) and the model is trained and tested multiple times on different folds.

**How It Works:**
1. Split the dataset into k folds like 5 or 10.
2. Train the model on (k-1) folds and test it on the remaining fold.
3. Repeat this process k times so that every fold is used for testing once.
4. Take the average of all results as the final performance score.

This process aims to accomplish the following:
1. Prevent overfitting during training by avoiding training and testing on the same subset of the data points.
2. Avoid information loss by using a certain subset of the data for validation only. This is important for small datasets.

Cross-validation is always good to be used for small datasets, and if used for large datasets the computational complexity will increase depending on the number of folds.

### Diagram
![Alt_text](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Figures/cross%20validation.png)

## Similar Question: Explain k-Fold Cross-Validation, Leave-One-Out (LOO) and Hold-Out Method.

### Answer
**k-Fold Cross-Validation:** The dataset is divided into k equal folds. The model is trained on (k-1) folds and tested on the remaining fold. This process is repeated k times, with each fold used once as the test set. The final score is the average of all k test results.

**Leave-One-Out Cross-Validation (LOO):** A special case of k-Fold where k = number of samples. Each observation is used once as the test set while the remaining data is used for training. It gives very accurate estimates but is computationally expensive for large datasets.

**Hold-Out Method:** The simplest technique where the dataset is split into two parts: a training set and a testing set (e.g., 70% train, 30% test). The model is trained on the training set and evaluated on the test set. It is fast but may lead to biased results depending on the split.

### Formula
CV error = (1/k)∑errorᵢ

## Similar Question: What is stratified cross-validation and when should we use it?

### Answer
Cross-validation is a technique for dividing data between training and validation sets. On typical cross-validation this split is done randomly. But in stratified cross-validation, the split preserves the ratio of the categories on both the training and validation datasets.

### Example
If we have a dataset with 10% of category A and 90% of category B, and we use stratified cross-validation, we will have the same proportions in training and validation. In contrast, if we use simple cross-validation, in the worst case we may find that there are no samples of category A in the validation set.

Stratified cross-validation may be applied in the following scenarios:
- On a dataset with multiple categories. The smaller the dataset and the more imbalanced the categories, the more important it will be to use stratified cross-validation.
- On a dataset with data of different distributions. For example, in a dataset for autonomous driving, we may have images taken during the day and at night. If we do not ensure that both types are present in training and validation, we will have generalization problems.

---

# Feature Engineering

## Primary Question: What is Feature Engineering in Machine Learning?

### Answer
Feature engineering is the process of creating, transforming or selecting relevant features from raw data to improve the performance of a machine learning model. It also reduces overfitting and makes the model easier to interpret.

Key Steps in Feature Engineering:
- Feature Creation: Generate new features from existing data like extracting "year" or "month" from a date column.
- Feature Transformation: Apply scaling, normalization or mathematical transformations (log, square root) to features.
- Feature Encoding: Convert categorical variables into numerical form like one-hot encoding, label encoding.
- Feature Selection: Identify and keep only the most relevant features using techniques like correlation analysis, mutual information or model-based importance scores.

### Example
Raw data: Date of Birth → Feature engineered: Age. Raw data: Text review → Feature engineered: Sentiment score.

## Similar Question: Difference between Feature Engineering and Feature Selection?

### Answer
**Feature Engineering**
- Feature Engineering is the process of creating new features or transforming existing features from raw data.
- It helps models capture hidden patterns and relationships in the data.
- Common techniques include encoding categorical variables, extracting date and time features, handling missing values, creating interaction features, and applying mathematical transformations.
- It requires domain knowledge and creativity to design meaningful features.
- Goal: Improve the quality and predictive power of the dataset by generating informative features.

**Feature Selection**
- Feature Selection is the process of selecting the most relevant features while removing irrelevant, redundant, or noisy ones.
- It reduces the dimensionality of the dataset, making models simpler and faster to train.
- It helps prevent overfitting and improves model interpretability.
- Common methods include Filter Methods (Chi-Square, Correlation), Wrapper Methods (Recursive Feature Elimination), and Embedded Methods (Lasso, Feature Importance).
- Goal: Retain only the most useful features for building an efficient and accurate model.

---

# Categorical Data Handling

## Primary Question: What is Categorical Data and how to handle it?

### Answer
Categorical data refers to features that represent discrete values or categories, rather than continuous numerical values. Examples include gender (Male, Female), color (Red, Blue, Green) or product type (Electronics, Clothing).

Types of Categorical Data:
- Nominal: Categories with no inherent order. Example: Red, Blue, Green.
- Ordinal: Categories with a meaningful order. Example: Low, Medium, High.

Machine learning models require numerical inputs, so categorical data needs to be handled using encoding. Common techniques include:
1. Label Encoding: Converts each category into a unique integer. Example: Red=0, Blue=1, Green=2. Suitable for ordinal data.
2. One-Hot Encoding: Converts categories into binary vectors, creating a new column for each category. Example: Color with Red, Blue, Green becomes three columns: [1,0,0], [0,1,0], [0,0,1]. Suitable for nominal data.
3. Binary Encoding: Converts categories into binary code, reducing dimensionality compared to one-hot encoding.
4. Target / Mean Encoding: Replaces categories with mean of target variable for regression or probability of positive class in classification.

---

# Label Encoding vs One-Hot Encoding

## Primary Question: When to use a Label Encoding vs. One Hot Encoding?

### Answer
This question generally depends on your dataset and the model which you wish to apply. But still, a few points to note before choosing the right encoding technique for your model:

We apply One-Hot Encoding when:
- The categorical feature is not ordinal (like the countries above)
- The number of categorical features is less so one-hot encoding can be effectively applied

We apply Label Encoding when:
- The categorical feature is ordinal (like Jr. kg, Sr. kg, Primary school, high school)
- The number of categories is quite large as one-hot encoding can lead to high memory consumption

## Similar Question: Difference between label encoding and one hot encoding?

### Answer
**Label Encoding**
- Label Encoding assigns a unique numerical value to each category.
- Each category is represented by a single integer (e.g., Red = 0, Blue = 1, Green = 2).
- It is memory-efficient because it creates only one column.
- It is suitable for ordinal data, where categories have a meaningful order (e.g., Low, Medium, High).
- It may introduce an unintended ordinal relationship for nominal data, causing some models to interpret one category as greater than another.
- Goal: Convert categorical values into numerical labels while keeping the number of features unchanged.

**One-Hot Encoding**
- One-Hot Encoding creates a separate binary column for each category.
- Each category is represented by a vector containing 0s and 1s (e.g., Red = [1,0,0], Blue = [0,1,0], Green = [0,0,1]).
- It increases the number of features, especially when there are many unique categories.
- It is suitable for nominal data, where categories do not have any natural order.
- It avoids introducing false relationships between categories.
- Goal: Represent categorical data without implying any ordinal relationship.

---

# Upsampling/Downsampling and SMOTE (Imbalanced Data)

## Primary Question: What is an imbalanced dataset? Can you list some ways to deal with it?

### Answer
An imbalanced dataset is one that has different proportions of target categories. For example, a dataset with medical images where we have to detect some illness will typically have many more negative samples than positive samples—say, 98% of images are without the illness and 2% of images are with the illness.

There are different options to deal with imbalanced datasets:
- Oversampling or undersampling. Instead of sampling with a uniform distribution from the training dataset, we can use other distributions so the model sees a more balanced dataset.
- Data augmentation. We can add data in the less frequent categories by modifying existing data in a controlled way. In the example dataset, we could flip the images with illnesses, or add noise to copies of the images in such a way that the illness remains visible.
- Using appropriate metrics. In the example dataset, if we had a model that always made negative predictions, it would achieve a precision of 98%. There are other metrics such as precision, recall, and F-score that describe the accuracy of the model better when using an imbalanced dataset.

## Similar Question: You are building a binary classifier and you found that the data is imbalanced, what should you do to handle this situation?

### Answer
If there is a data imbalance there are several measures we can take to train a fairer binary classifier:

**1. Pre-Processing:**
- Check whether you can get more data or not.
- Use sampling techniques (Sample minority class, Downsample majority class, can take the hybrid approach as well). We can also use data augmentation to add more data points for the minority class but with little deviations/changes leading to new data points that are similar to the ones they are derived from. The most common/popular technique is SMOTE (Synthetic Minority Oversampling technique).
- Suppression: Though not recommended, we can drop off some features directly responsible for the imbalance.
- Learning Fair Representation: Projecting the training examples to a subspace or plane minimizes the data imbalance.
- Re-Weighting: We can assign some weights to each training example to reduce the imbalance in the data.

**2. In-Processing:**
- Regularisation: We can add score terms that measure the data imbalance in the loss function and therefore minimizing the loss function will also minimize the degree of imbalance concerning the score chosen which also indirectly minimizes other metrics that measure the degree of data imbalance.
- Adversarial Debiasing: Here we use the adversarial notion to train the model where the discriminator tries to detect if there are signs of data imbalance in the predicted data by the generator and hence the generator learns to generate data that is less prone to imbalance.

**3. Post-Processing:**
- Odds-Equalization: Here we try to equalize the odds for the classes with respect to the data is imbalanced for correct imbalance in the trained model. Usually, the F1 score is a good choice, if both precision and recall scores are important.
- Choose appropriate performance metrics. For example, accuracy is not a correct metric to use when classes are imbalanced. Instead, use precision, recall, F1 score, and ROC curve.

### Diagram
![Alt_text](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Figures/Oversampling.png)

## Similar Question: What is Upsampling and Downsampling?

### Answer
Upsampling and downsampling are techniques used to handle imbalanced datasets where the number of samples in different classes is unequal.

**Upsampling (Oversampling):** Increases the number of samples in the minority class to balance the dataset. Techniques include:
- Random Oversampling: Duplicate random samples from the minority class.
- SMOTE (Synthetic Minority Over-sampling Technique): Generate synthetic samples by interpolating between existing minority samples.

**Downsampling (Undersampling):** Reduces the number of samples in the majority class to balance the dataset. Techniques include:
- Random Undersampling: Randomly remove samples from the majority class.
- Cluster-Based Undersampling: Remove samples based on clustering to retain diversity.

### Example
We have a dataset of 1000 positive samples, 100 negative samples. Upsampling creates 900 additional negative samples. Downsampling reduces positive samples to 100.

## Similar Question: Explain SMOTE method used to handle data imbalance

### Answer
SMOTE (Synthetic Minority Over-sampling Technique) is an oversampling technique used to handle imbalanced datasets by increasing the number of samples in the minority class.

Instead of simply duplicating existing instances, SMOTE generates synthetic (new) data points by interpolating between a minority class sample and its nearest minority class neighbors. This helps the model learn better decision boundaries and reduces bias toward the majority class.

How SMOTE works:
1. Select a sample from the minority class.
2. Find its k-nearest minority class neighbors.
3. Randomly choose one of the neighbors.
4. Generate a new synthetic sample along the line connecting the two samples using linear interpolation.
5. Repeat the process until the desired level of class balance is achieved.

---

# Handling Missing/Duplicate/Corrupted Data

## Primary Question: How to handle missing and duplicate values?

### Answer
In general, real-world data often has a lot of missing values. The cause of missing values can be data corruption or failure to record data.

Missing values are common in real-world datasets and can affect model performance. Techniques to Handle Missing Values are:

1. Remove Rows or Columns:
   - Drop rows with missing values using dropna() in pandas.
   - Drop columns if most values are missing.
2. Imputation:
   - Mean/Median/Mode Imputation: Replace missing values with mean/median (for numerical) or mode (for categorical).
   - Forward/Backward Fill: Fill missing values using previous or next valid value in time series data.
   - Prediction-Based Imputation: Predict missing values using a model trained on other features.
3. Flag Missing Values:
   - Create a new binary column to indicate whether a value was missing.

Duplicate rows can lead to biased or misleading results.

---

# Outliers - What They Are and How to Handle

## Primary Question: What are outliers and how to handle them?

### Answer
Investigating the outliers is always the first step in understanding how to treat them. After you understand the nature of why the outliers occurred you can apply one of the several methods.

Outliers are data points that differ significantly from other observations in the dataset. They can arise due to errors, variability in data or rare events.
- Can skew statistics like mean and standard deviation.
- Can mislead machine learning models, especially regression and distance-based algorithms.

Detection Methods:
- Box Plot / IQR Method: Identify points outside Q1 - 1.5*IQR or Q3 + 1.5*IQR.
- Z-Score Method: Points with |z| > 3 are considered outliers.
- Visualization: Scatter plots, histograms or violin plots.

Handling Methods:
- Remove Outliers: Delete extreme values if they are errors or irrelevant.
- Transform Data: Apply log, square root or other transformations to reduce skewness.
- Cap/Floor Values: Replace extreme values with upper/lower bounds (Winsorization).
- Use Robust Models: Models like Decision Trees or Random Forests are less sensitive to outliers.

---

# Data Leakage

## Primary Question: Explain Data Leakage in Machine Learning.

### Answer
Data leakage occurs when information from outside the training dataset particularly information that wouldn't be available at prediction time is used to build the model.

Common Causes:
- Target Leakage: A feature that is influenced by or derived from the target variable is included in training data. Example: Including "days_since_diagnosis" as a feature when predicting whether a patient has a disease.
- Train-Test Contamination: Preprocessing steps like scaling, imputation, or feature selection are performed on the entire dataset before splitting into train/test, letting test-set statistics leak into training.
- Temporal Leakage: In time series data, using future information to predict the past — e.g., shuffling time-ordered data before splitting.
- Duplicate Records: The same or near-identical records appearing in both train and test sets.

How to Prevent It:
- Always split data into train/test before any preprocessing, and fit scalers/encoders only on the training set.
- For time series, use time-based splits rather than random splits.
- Carefully audit features for any that wouldn't realistically be available at prediction time.
- Use pipelines (e.g., scikit-learn's Pipeline) to enforce that transformations are fit only on training folds during cross-validation.

---

# Different Hypothesis in ML

## Primary Question: Different Hypothesis in Machine Learning?

### Answer
In machine learning, a hypothesis is a function or model that maps input features to output predictions. Different hypotheses represent different types of models or assumptions about the data.

1. Null Hypothesis (H₀): Assumes no effect or no relationship exists between features and target. Often used in statistical testing to validate model assumptions. Example: "Feature X has no impact on predicting Y."
2. Alternative Hypothesis (H₁ or Ha): Assumes there is a relationship or effect. Example: "Feature X significantly affects predicting Y."
3. Parametric Hypotheses: Assume the data follows a known distribution and have fixed parameters. Example: Linear regression assumes a linear relationship with parameters (weights).
4. Non-Parametric Hypotheses: Make no assumptions about the underlying data distribution. Examples: Decision Trees, K-Nearest Neighbors.
5. Machine Learning Hypothesis Function (hθ): Represents the model used to make predictions. In supervised learning, the goal is to find the hypothesis that minimizes error on the training data.

### Formula
hθ(x) = θ₀ + θ₁x₁ + θ₂x₂ + ⋯ + θₙxₙ

---

# Hyperparameter Tuning

## Primary Question: What is Hyperparameter Tuning in Machine Learning?

### Answer
Hyperparameter tuning is the process of finding the best set of hyperparameters for a machine learning model to maximize performance. Hyperparameters are parameters set before training like learning rate, number of trees in Random Forest, regularization strength, etc that cannot be learned directly from the data.

Common Hyperparameter Tuning Methods are:
- Grid Search: It tries all possible combinations of hyperparameters from a predefined set. It is simple and exhaustive, but computationally expensive for large search spaces.
- Random Search: Randomly samples combinations of hyperparameters are taken from a given range. Often faster than grid search and can find good results in fewer iterations.
- Bayesian Optimization: Builds a probabilistic model of the objective function and selects hyperparameters to maximize performance. Efficient for expensive models; balances exploration and exploitation.

---

# Linear Regression and Assumptions

## Primary Question: What is Linear Regression? What are its Assumption?

### Answer
Linear regression is a supervised statistical model to predict dependent variable quantity based on independent variables. Linear regression is a parametric model and the objective of linear regression is that it has to learn coefficients using the training data and predict the target value given only independent values.

Linear Regression is a supervised learning algorithm used to predict a continuous target variable based on one or more input features by fitting a linear relationship.

Some of the linear regression assumptions and how to validate them:
1. Linear relationship between independent and dependent variables
2. Independent residuals and the constant residuals at every x. We can check for 1 and 2 by plotting the residuals(error terms) against the fitted values (upper left graph). Generally, we should look for a lack of patterns and a consistent variance across the horizontal line.
3. Normally distributed residuals. We can check for this using a couple of methods:
   - Q-Q-plot(upper right graph): If data is normally distributed, points should roughly align with the 45-degree line.
   - Boxplot: it also helps visualize outliers.
   - Shapiro–Wilk test: If the p-value is lower than the chosen threshold, then the null hypothesis (Data is normally distributed) is rejected.
4. Low multicollinearity: you can calculate the VIF (Variable Inflation Factors) using your favorite statistical tool. If the value for each covariate is lower than 10 (some say 5), you're good to go.

Assumptions of Linear Regression (summary):
- Linearity: Relationship between x and y is linear.
- Independence: Data points are independent.
- Homoscedasticity: Error terms have constant variance.
- Normality of Errors: Residuals follow a normal distribution.
- No Multicollinearity: Features should not be highly correlated.

### Formula
y = mx + c

Where: y = predicted output, x = input feature, m = slope (coefficient), c = intercept

### Diagram
![alt text](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Figures/Linear%20regression%20assumptions.jpg)

---

# Logistic Regression

## Primary Question: Explain briefly the logistic regression model and state an example of when you have used it recently?

### Answer
Logistic regression is used to calculate the probability of occurrence of an event in the form of a dependent output variable based on independent input variables. Logistic regression is commonly used to estimate the probability that an instance belongs to a particular class. If the probability is bigger than 0.5 then it will belong to that class (positive) and if it is below 0.5 it will belong to the other class. This will make it a binary classifier.

It is important to remember that the Logistic regression isn't a classification model, it's an ordinary type of regression algorithm, and it was developed and used before machine learning, but it can be used in classification when we put a threshold to determine specific categories.

### Example
There is a lot of classification applications to it: Classify email as spam or not, To identify whether the patient is healthy or not, and so on.

## Similar Question: Explain how sigmoid function work in Logistic Regression and why it is not a Regression Model even though its name has it?

### Answer
In logistic regression, we want to predict probabilities for binary outcomes (e.g., 0 or 1). The sigmoid function converts any real number into a value between 0 and 1, making it suitable for probabilities.

Despite the name, logistic regression is used for classification, not regression. As it does not predict continuous values like linear regression, but instead predicts probabilities. A threshold (e.g., 0.5) is applied to classify outcomes as 0 or 1.

### Formula
σ(z) = 1 / (1 + e⁻ᶻ)

---

# Multicollinearity and VIF

## Primary Question: What is Multicollinearity and Why is it a Problem?

### Answer
Multicollinearity occurs when two or more independent features are highly correlated with each other in a dataset. This means one feature can be linearly predicted from another with high accuracy. It can cause problems like:
- Unstable Coefficients: Makes regression coefficients unreliable and highly sensitive to small changes in data.
- Interpretation Difficulty: Hard to determine the individual effect of each feature on the target variable.
- Reduced Model Performance: May not affect prediction accuracy much, but impacts the explainability of the model.
- Inflated Variance: Leads to high standard errors in coefficient estimates.

Detection Methods:
- Correlation Matrix: Check for high correlation between features.
- Variance Inflation Factor (VIF): A VIF > 5 (or 10) usually indicates strong multicollinearity.

Solution:
- Remove or combine correlated features.
- Use Principal Component Analysis (PCA) or other dimensionality reduction techniques.
- Apply regularization methods like Ridge regression.

## Similar Question: What is Variance Inflation Factor?

### Answer
The Variance Inflation Factor (VIF) is a statistical measure used to detect multicollinearity in regression models. It shows how much the variance of a regression coefficient is inflated because of correlation with other independent variables.

Interpretation:
- VIF = 1: No correlation with other features.
- 1 < VIF < 5: Moderate correlation, usually acceptable.
- VIF > 5 (or 10): High multicollinearity i.e it is problematic and needs to be resolved.

### Formula
VIFᵢ = 1 / (1 − Rᵢ²)

Here Rᵢ² is the coefficient of determination when the i-th feature is regressed on all other features.

---

# Information Gain and Entropy in Decision Trees

## Primary Question: Explain what is information gain and entropy in the context of decision trees?

### Answer
Entropy and Information Gain are two key metrics used in determining the relevance of decision-making when constructing a decision tree model and determining the nodes and the best way to split.

The idea of a decision tree is to divide the data set into smaller data sets based on the descriptive features until we reach a small enough set that contains data points that fall under one label.

**Entropy** is the measure of impurity, disorder, or uncertainty in a bunch of examples. Entropy controls how a Decision Tree decides to split the data. If all samples belong to the same class → entropy = 0 (pure). If classes are equally mixed → entropy is high.

**Information gain** calculates the reduction in entropy or surprise from transforming a dataset in some way. It is commonly used in the construction of decision trees from a training dataset, by evaluating the information gain for each variable and selecting the variable that maximizes the information gain, which in turn minimizes the entropy and best splits the dataset into groups for effective classification.

Relationship between Entropy and Information Gain:
- Entropy should be minimized i.e less impurity after splitting.
- Information Gain should be maximized i.e best feature chosen for splitting.
- Decision Trees always pick the feature with the highest Information Gain to reduce entropy the most.

### Formula
Entropy(S) = −∑pᵢlog₂(pᵢ), where pᵢ = proportion of samples belonging to class i, c = number of classes

IG(S,A) = Entropy(S) − ∑ (|Sᵥ|/|S|) Entropy(Sᵥ), for v ∈ Values(A)

Where S = dataset, A = feature used for split, Sᵥ = subset of S where feature A = v

---

# Decision Trees - Splitting, Pruning, Algorithms

## Primary Question: Explain ID3 and CART

### Answer
**ID3 (Iterative Dichotomiser 3):** ID3 is a decision tree algorithm used only for classification. It uses Entropy and Information Gain to decide which feature should split the dataset. It works like:
1. Calculate entropy of the dataset.
2. For each feature, calculate information gain.
3. Choose the feature with the highest information gain for the split.
4. Repeat this process until all records are classified or stopping conditions are met.

**CART (Classification and Regression Trees):** CART can be used for both classification and regression problems. It uses Gini Index for classification and Mean Squared Error (MSE) for regression. It works like:
1. For each feature, calculate Gini Index (for classification) or MSE (for regression).
2. Select the feature with the lowest impurity.
3. Split the dataset into two branches (CART always creates binary trees).
4. Repeat the process until stopping conditions are met.

## Similar Question: What are the different methods to split a tree in a decision tree algorithm?

### Answer
Decision trees can be of two types regression and classification. For classification, classification accuracy created a lot of instability. So the following loss functions are used:

- Gini's Index: Gini impurity is used to predict the likelihood of a randomly chosen example being incorrectly classified by a particular node. It's referred to as an "impurity" measure because it demonstrates how the model departs from a simple division.
- Cross-entropy or Information Gain: Information gain refers to the process of identifying the most important features/attributes that convey the most information about a class. The entropy principle is followed with the goal of reducing entropy from the root node to the leaf nodes. Information gain is the difference in entropy before and after splitting, which describes the impurity of in-class items.

For regression, the good old mean squared error serves as a good loss function which is minimized by splits of the input features and predicting the mean value of the target feature on the subspaces resulting from the split. But finding the split that results in the minimum possible residual sum of squares is computationally infeasible, so a greedy top-down approach is taken i.e. the splits are made at a level from top to down which results in maximum reduction of RSS. We continue this until some maximum depth or number of leaves is attained.

## Similar Question: How to Prevent Overfitting in Decision Trees?

### Answer
Decision trees are prone to overfitting because they can grow very deep and capture noise along with patterns. To prevent overfitting we can use following techniques:
- Limit Tree Depth: Restrict max_depth so the tree doesn't grow too complex.
- Minimum Samples for Split/Leaf: Set min_samples_split or min_samples_leaf to ensure splits happen only when enough data is present.
- Pruning: Remove branches that add little value (pre-pruning or post-pruning).
- Feature Selection: Use only relevant features to avoid unnecessary splits.
- Use Ensemble Methods: Techniques like Random Forest and Gradient Boosting average multiple trees to reduce variance.
- Cross-Validation: Helps monitor performance on unseen data and avoid overly complex trees.

## Similar Question: What is Pruning in Decision Trees?

### Answer
Pruning is the process of removing unnecessary branches from a decision tree that do not provide significant predictive knowledge. It improves generalization on unseen data and makes the model more interpretable and efficient. We have 2 types of pruning:

1. Pre-Pruning (Early Stopping): Stop growing the tree early by setting constraints like max_depth, min_samples_split or min_samples_leaf. Prevents the tree from becoming too complex.
2. Post-Pruning: Grow the full tree first, then remove branches that have little impact on accuracy. Example: Cost Complexity Pruning (CCP) balances tree accuracy and size using a penalty term.

---

# Naive Bayes

## Primary Question: Explain Naive Bayes and Bayes' Theorem.

### Answer
Bayes' Theorem calculates the probability of an event based on prior knowledge of related events.

Where: P(A|B): Posterior probability (probability of A given B). P(B|A): Likelihood (probability of B given A). P(A): Prior probability of A. P(B): Evidence or probability of B.

Naive Bayes is a classification algorithm based on Bayes' theorem. It assumes that all features are independent (naive assumption). It is widely used in weather forecast and classifying emails as spam or not spam. Its working is:
1. Calculate prior probability of each class.
2. Compute likelihood of features for each class.
3. Apply Bayes' theorem to get posterior probability.
4. Assign the class with the highest posterior probability.

### Formula
P(A|B) = [P(B|A) · P(A)] / P(B)

## Similar Question: What are the assumptions of Naive Bayes?

### Answer
Naive Bayes is based on a few key assumptions that simplify calculations:
- Feature Independence: All features are assumed to be independent of each other given the class label.
- All Features Contribute Equally: Each feature contributes equally and independently to the outcome.
- Categorical or Conditional Probability: Features can be categorical or continuous, but for continuous data, it's assumed to follow a probability distribution (like Gaussian).
- Correctly Labeled Data: The training dataset is assumed to be accurately labeled, because incorrect labels affect probability estimates.

## Similar Question: What are the types of Naive Bayes algorithm?

### Answer
The main types of Naive Bayes algorithms are:
- Gaussian Naive Bayes: Assumes continuous features follow a normal (Gaussian) distribution. Often used for features like height, weight or temperature.
- Multinomial Naive Bayes: Works with discrete count data such as word counts in text classification (e.g., spam detection).
- Bernoulli Naive Bayes: Suitable for binary features (0 or 1) like whether a word occurs in a document or not.
- Categorical Naive Bayes: Used when features are categorical (like color: red, green, blue) rather than numeric.

---

# Generative vs Discriminative Models

## Primary Question: What's the difference between a generative and discriminative model?

### Answer
A generative model will learn categories of data while a discriminative model will simply learn the distinction between different categories of data. Discriminative models will generally outperform generative models on classification tasks.

**Generative Models**
- Learn the joint probability distribution P(X, Y) — i.e., how the data is generated for each class.
- Can generate new synthetic data samples similar to the training data.
- Model both the features and how classes produce those features.
- Examples: Naive Bayes, Gaussian Mixture Models (GMM), Hidden Markov Models (HMM).

**Discriminative Models**
- Learn the conditional probability P(Y | X) directly — i.e., the decision boundary between classes.
- Cannot generate new data; only classify given inputs.
- Generally achieve higher classification accuracy when there's enough training data, since they focus directly on the decision boundary rather than modeling the full data distribution.
- Examples: Logistic Regression, SVM, Decision Trees, Random Forest.

---

# K-Nearest Neighbors (KNN)

## Primary Question: Explain K-Nearest Neighbors (KNN) working.

### Answer
K-Nearest Neighbors (KNN) is a supervised learning algorithm used for classification and regression. It predicts the output of a data point based on the majority class or average value of its K nearest neighbors.

How KNN works:
1. Choose the number of neighbors K.
2. Calculate the distance (e.g., Euclidean) between the new data point and all points in the training set.
3. Select the K nearest neighbors based on distance.
4. For classification, assign the most frequent class among neighbors. For regression, take the average value of neighbors.

## Similar Question: Why is KNN a lazy algorithm?

### Answer
K-Nearest Neighbors (KNN) is called a lazy learning algorithm because it does not learn an explicit model during training. Instead, it stores all training data and waits until a query (test data) is given to make predictions.
- No training phase and all computation happens at prediction time.
- Memory-intensive because it stores the entire training dataset.
- Simplicity makes it easy to implement but can be slow for large datasets.

## Similar Question: How does the K value affect KNN?

### Answer
A small K value makes KNN sensitive to noise and can lead to overfitting. A large K value smooths decision boundaries but can cause underfitting by ignoring local patterns. Choosing the right K balances overfitting and underfitting, often determined using cross-validation.

## Similar Question: How to find the optimal value of K in KNN?

### Answer
Different techniques to find the optimal K include:
- Cross-Validation: Test multiple K values using k-fold cross-validation and pick the one with the best validation performance.
- Elbow Method: Plot error rate versus K and choose the K at the "elbow" point where error stabilizes.
- Silhouette Score: Evaluate clustering quality (for KNN in clustering contexts) to select K with the best score.
- Gap Statistic: Compare intra-cluster variation with a reference null distribution to choose K.
- Grid Search: Systematically test a range of K values and select the best based on performance metrics.
- Trial and Error: Manually test different K values and evaluate model accuracy (useful for small datasets).

## Similar Question: What is KNN Imputer and how does it work?

### Answer
KNN Imputer fills missing values by referencing the k nearest neighbors of a data point based on a distance metric (e.g., Euclidean distance).
- Neighborhood-based Imputation: Finds k closest points to the missing value.
- Imputation: Uses the mean or median of neighbors to fill the missing value.
- Distance Parameter: k defines how many neighbors are considered and the distance metric controls similarity.

---

# Distance Metrics in ML

## Primary Question: What are the different distance metrics in Machine Learning?

### Answer
Distance metrics measure how similar or dissimilar two data points are. They are widely used in clustering, K-NN and other ML algorithms. Different metrics work better depending on the type of data and problem. Common Distance Metrics:

1. Euclidean Distance: Straight-line distance between two points in space. Most commonly used for continuous data.
2. Manhattan Distance (L1 Norm): Sum of absolute differences between coordinates. Works well when you want to penalize differences equally across dimensions.
3. Minkowski Distance: Generalization of Euclidean and Manhattan distances. Can be tuned with a parameter p: p=1 → Manhattan, p=2 → Euclidean.
4. Cosine Similarity / Cosine Distance: Measures the angle between two vectors, not magnitude. Useful for text data, document similarity or high-dimensional sparse data.
5. Jaccard Distance: Measures dissimilarity between sets. Often used for binary or categorical features.

---

# Support Vector Machines (SVM)

## Primary Question: Explain the kernel trick in SVM and why we use it and how to choose what kernel to use?

### Answer
Kernels are used in SVM to map the original input data into a particular higher dimensional space where it will be easier to find patterns in the data and train the model with better performance.

Typically without the kernel trick, in order to calculate support vectors and support vector classifiers, we need first to transform data points one by one to the higher dimensional space, do the calculations based on SVM equations in the higher dimensional space, and then return the results. The 'trick' in the kernel trick is that we design the kernels based on some conditions as mathematical functions that are equivalent to a dot product in the higher dimensional space without even having to transform data points to the higher dimensional space. i.e. we can calculate support vectors and support vector classifiers in the same space where the data is provided which saves a lot of time and calculations.

Having domain knowledge can be very helpful in choosing the optimal kernel for your problem, however, in the absence of such knowledge following this default rule can be helpful: For linear problems, we can try linear or logistic kernels, and for nonlinear problems, we can use RBF or Gaussian kernels.

The kernel trick in SVM is a technique that allows the algorithm to handle non-linear data by transforming it into a higher-dimensional space where it becomes linearly separable. Instead of explicitly computing the transformation, the kernel function computes the similarity between data points in the transformed space hence making the process efficient.

Popular kernel functions in SVM:
- Linear Kernel: Works well when the data is linearly separable or when the number of features is very large like text classification. It is fast and simple.
- Polynomial Kernel: Useful when the relationship between features is non-linear but still polynomial in nature. Example: classifying data that follows circular or curved boundaries.
- RBF (Radial Basis Function) Kernel / Gaussian Kernel: Best for highly complex and non-linear data where no clear linear boundary exists. It is the most commonly used kernel in practice.
- Sigmoid Kernel: Behaves similarly to neural networks and can be used for certain datasets, but it is less common compared to RBF and polynomial kernels.

### Example
If we have binary class data which form a ring-like pattern (inner and outer rings representing two different class instances) when plotted in 2D space, a linear SVM kernel will not be able to differentiate the two classes well when compared to an RBF (radial basis function) kernel, mapping the data into a particular higher dimensional space where the two classes are clearly separable.

### Diagram
![Alt_text](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Figures/Kerenl%20trick.png)

## Similar Question: What is the decision boundary in SVM?

### Answer
In Support Vector Machine (SVM), the decision boundary is the line (in 2D) or hyperplane (in higher dimensions) that separates data points of different classes. It is chosen so that the margin which is the distance between the hyperplane and the nearest data points, called support vectors is maximized. The decision boundary in SVM is the hyperplane that best separates the classes while maintaining the largest possible margin for better generalization.

## Similar Question: Does SVM only work with linear data points?

### Answer
No, SVMs are not limited to linear data. While a linear SVM works well when data is linearly separable, for non-linear data SVM uses the kernel trick. Kernels like polynomial, RBF or sigmoid transform the data into a higher-dimensional space where a linear separation becomes possible.

## Similar Question: Do you need to scale your data if you will be using the SVM classifier and discuss your answer

### Answer
Yes, feature scaling is required for SVM and all margin-based classifiers since the optimal hyperplane (the decision boundary) is dependent on the scale of the input features. In other words, the distance between two observations will differ for scaled and non-scaled cases, leading to different models being generated.

This can be seen in the figure below, when the features have different scales, we can see that the decision boundary and the support vectors are only classifying the X1 features without taking into consideration the X0 feature, however after scaling the data to the same scale the decision boundaries and support vectors are looking much better and the model is taking into account both features.

To scale the data, normalization, and standardization are the most popular approaches.

### Diagram
![SVM scaled Vs non scaled](https://user-images.githubusercontent.com/72076328/192571498-4a939472-7bb1-4bf2-963f-a6e6394802ba.png)

---

# Random Forest

## Primary Question: What is Random Forest?

### Answer
Random Forest is an ensemble learning method that builds multiple decision trees and combines their results to improve accuracy and stability. Instead of relying on a single decision tree, it takes the majority vote (for classification) or average (for regression) of many trees.
- Handles large datasets with higher accuracy than a single decision tree.
- Reduces overfitting by combining multiple trees.
- Works well with both categorical and numerical data.
- Provides feature importance, helping to understand which variables are most influential.

How Random Forest Works:
1. Creates multiple random subsets of the dataset using bootstrapping (sampling with replacement).
2. Builds a decision tree for each subset, but at each node, it selects a random subset of features instead of using all features.
3. Each tree makes a prediction independently.
4. The final prediction is made by combining all tree outputs (majority voting for classification, average for regression).

## Similar Question: Describe the motivation behind random forests and mention two reasons why they are better than individual decision trees?

### Answer
The motivation behind random forest or ensemble models in general in layman's terms, Let's say we have a question/problem to solve we bring 100 people and ask each of them the question/problem and record their solution — combining many diverse opinions (trees) tends to give a better, more robust answer than relying on a single individual's (tree's) judgment.

## Similar Question: What are some of the hyperparameters of the random forest regressor which help to avoid overfitting?

### Answer
The important hyperparameters of a Random Forest Regressor that help to control overfitting are:
- max_depth: Restricts how deep a tree can grow. Smaller depth reduces complexity and prevents overfitting.
- n_estimators: Number of trees in the forest. More trees usually improve stability, but too many just increase computation without reducing overfitting.
- min_samples_split: Minimum samples required to split a node. Higher values prevent trees from creating overly specific splits.
- min_samples_leaf: Minimum samples required at a leaf node. Larger values create smoother predictions and avoid capturing noise.
- max_leaf_nodes: Limits the number of leaf nodes, thereby controlling tree growth and depth.
- max_features: Number of features considered for splitting at each node. Using fewer features introduces randomness and helps reduce overfitting.
- bootstrap: Whether to sample data with replacement for each tree. Bootstrapping introduces diversity in trees, reducing overfitting.
- max_samples: If bootstrap is True, this defines how many samples are drawn to train each tree. Controlling this adds more randomness.

## Similar Question: Whether decision tree or random forest is more robust to outliers

### Answer
Decision trees are somewhat sensitive to outliers, as extreme values can influence the splits. Random forests, being an ensemble of multiple decision trees, aggregate results from several trees which reduces the impact of outliers. Therefore, random forests are generally more robust to outliers compared to a single decision tree.

## Similar Question: How does Random Forest ensure diversity among trees?

### Answer
- Bagging (Bootstrap Aggregating): Each tree is trained on a random subset of the data.
- Feature Randomness: Each split considers a random subset of features, preventing trees from being identical.

### Quick Explanation (AI Generated)
Bootstrapping (referenced above) is a sampling technique where multiple datasets are created by randomly selecting data points with replacement from the original dataset — the same data point can appear multiple times in a new sample, and each bootstrap sample is usually the same size as the original dataset. It is used in ensemble methods like Bagging and Random Forest to reduce variance and prevent overfitting.

---

# Gradient Boosting vs Random Forest

## Primary Question: What are the differences and similarities between gradient boosting and random forest? and what are the advantages and disadvantages of each when compared to each other?

### Answer
Similarities:
1. Both these algorithms are decision-tree-based algorithms
2. Both these algorithms are ensemble algorithms
3. Both are flexible models and do not need much data preprocessing.

Differences (as summarized elsewhere in the notes): Random Forest builds trees independently in parallel using bagging and averages/votes their results, reducing variance and being more robust to overfitting; Gradient Boosting builds trees sequentially using boosting, where each new tree corrects the errors of the previous ones, reducing bias but being more prone to overfitting if not carefully tuned.

---

# Boosting Algorithms (AdaBoost, XGBoost, CatBoost)

## Primary Question: Explain AdaBoost, XGBoost and CatBoost.

### Answer
**AdaBoost (Adaptive Boosting)**
- Works by combining multiple weak learners (usually shallow decision trees).
- Each new learner focuses more on the misclassified samples of the previous learners by assigning higher weights to them.
- Final prediction is a weighted majority vote (classification) or weighted sum (regression).
- Best used when: You have simple models and want to improve them by focusing on difficult cases.

**XGBoost (Extreme Gradient Boosting)**
- An optimized implementation of Gradient Boosting with high performance and regularization features.
- Builds trees sequentially where each new tree corrects errors of the previous ones by minimizing a differentiable loss function.
- Includes regularization terms (L1 and L2) to avoid overfitting.
- Highly efficient, parallelizable and widely used in competitions.
- Best used when: You need fast, accurate models for structured/tabular data.

**CatBoost (Categorical Boosting)**
- A gradient boosting algorithm designed to handle categorical features directly without needing extensive preprocessing like one-hot encoding.
- Uses ordered boosting to reduce prediction shift (bias caused by using the same data for building and training).
- Often requires less tuning and works well.
- Best used when: Dataset has many categorical features and minimal preprocessing is preferred.

One way for a new predictor to correct its predecessor is to pay a bit more attention to the training instances that the predecessor under-fitted (Adaptive Boosting). This results in new predictors focusing more and more on the hard cases. Just like AdaBoost, Gradient Boosting works by sequentially adding predictors to an ensemble, each one correcting its predecessor. However, instead of tweaking the instance weights at every iteration as AdaBoost does, this method tries to fit the new predictor to the residual errors made by the previous predictor.

### Diagram
![1661788022018](https://user-images.githubusercontent.com/72076328/187241588-6cc3166f-a3e0-46b9-a0ce-e3d9ef9f0228.jpg)

## Similar Question: What is the difference between Gradient Boosting and CatBoost?

### Answer
**Gradient Boosting**
- Gradient Boosting builds decision trees sequentially, where each new tree corrects the errors made by the previous trees.
- It minimizes the prediction error by optimizing a loss function using gradient descent.
- It requires manual preprocessing of categorical features, such as Label Encoding or One-Hot Encoding.
- It is prone to overfitting if not properly tuned.
- It generally requires careful hyperparameter tuning for optimal performance.
- Goal: Improve prediction accuracy by combining multiple weak learners into a strong learner.

**CatBoost**
- CatBoost (Categorical Boosting) is a Gradient Boosting algorithm developed by Yandex.
- It can handle categorical features directly without requiring manual encoding.
- It uses ordered boosting to reduce prediction bias and minimize overfitting.
- It automatically processes categorical variables and missing values efficiently.
- It often delivers high accuracy with minimal hyperparameter tuning.
- Goal: Provide a faster, more accurate Gradient Boosting algorithm, especially for datasets with many categorical features.

---

# Ensemble Learning - Bagging vs Boosting

## Primary Question: What's the difference between boosting and bagging?

### Answer
Boosting and bagging are similar, in that they are both ensembling techniques, where a number of weak learners (classifiers/regressors that are barely better than guessing) combine (through averaging or max vote) to create a strong learner that can make accurate predictions. Bagging means that you take bootstrap samples (with replacement) of your data set and each sample trains a (potentially) weak learner. Boosting, on the other hand, uses all data to train each learner, but instances that were misclassified by the previous learners are given more weight so that subsequent learners give more focus to them during training.

**Bagging (Bootstrap Aggregating):**
- Bagging trains multiple models in parallel using random subsets of the training data (with replacement).
- Each model gives a prediction and results are combined by majority voting (classification) or averaging (regression).
- It reduces variance and prevents overfitting.
- Example: Random Forest.

**Boosting:**
- Boosting trains models sequentially where each new model focuses on correcting the mistakes of the previous ones.
- Models are combined by assigning higher weights to more accurate models.
- It reduces bias and improves accuracy but can overfit if not controlled.
- Examples: AdaBoost, Gradient Boosting, XGBoost.

## Similar Question: What is boosting in the context of ensemble learners, discuss two famous boosting methods

### Answer
Boosting refers to any Ensemble method that can combine several weak learners into a strong learner. The general idea of most boosting methods is to train predictors sequentially, each trying to correct its predecessor.

There are many boosting methods available, but by far the most popular are:
- Adaptive Boosting: One way for a new predictor to correct its predecessor is to pay a bit more attention to the training instances that the predecessor under-fitted. This results in new predictors focusing more and more on the hard cases.
- Gradient Boosting: Another very popular Boosting algorithm is Gradient Boosting. Just like AdaBoost, Gradient Boosting works by sequentially adding predictors to an ensemble, each one correcting its predecessor. However, instead of tweaking the instance weights at every iteration as AdaBoost does, this method tries to fit the new predictor to the residual errors made by the previous predictor.

## Similar Question: Why is boosting a more stable algorithm as compared to other ensemble algorithms?

### Answer
Boosting algorithms focus on errors found in previous iterations until they become obsolete. Whereas in bagging there is no corrective loop. That's why boosting is a more stable algorithm compared to other ensemble algorithms.

---

# Why Ensembles Perform Better

## Primary Question: Why do ensembles typically have higher scores than individual models?

### Answer
An ensemble is the combination of multiple models to create a single prediction. The key idea for making better predictions is that the models should make different errors. That way the errors of one model will be compensated by the right guesses of the other models and thus the score of the ensemble will be higher.

We need diverse models for creating an ensemble. Diversity can be achieved by:
- Using different ML algorithms. For example, you can combine logistic regression, k-nearest neighbors, and decision trees.
- Using different subsets of the data for training. This is called bagging.
- Giving a different weight to each of the samples of the training set. If this is done iteratively, weighting the samples according to the errors of the ensemble, it's called boosting.

Many winning solutions to data science competitions are ensembles. However, in real-life machine learning projects, engineers need to find a balance between execution time and accuracy.

---

# Voting Classifiers (Hard vs Soft)

## Primary Question: What is the difference between hard and soft voting classifiers in the context of ensemble learners?

### Answer
- Hard Voting: We take into account the class predictions for each classifier and then classify an input based on the maximum votes to a particular class.
- Soft Voting: We take into account the probability predictions for each class by each classifier and then classify an input to the class with maximum probability based on the average probability (averaged over the classifier's probabilities) for that class.

### Diagram
![Alt_text](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Figures/Hard%20Vs%20soft%20voting.png)

---

# K-Means Clustering

## Primary Question: Explain K-Means Clustering

### Answer
K-Means is a well-known clustering algorithm. K-means clustering is often used because it is easy to interpret and implement.

Clustering is an unsupervised learning technique where data is grouped into clusters such that:
- Points in the same cluster are more similar to each other.
- Points in different clusters are more dissimilar from each other.

K-Means is a popular clustering algorithm that divides the dataset into K clusters. Each cluster is represented by its centroid (average of all data points in that cluster).
- It works well when clusters are spherical and well-separated.
- Sensitive to the initial placement of centroids.
- Requires predefining K which can be chosen using methods like Elbow Method or Silhouette Score.
- Can struggle with non-linear or overlapping clusters.

How K-Means Works:
1. Choose the number of clusters K.
2. Randomly initialize K centroids.
3. Assign each data point to the nearest centroid (cluster assignment).
4. Recalculate centroids as the mean of all points in a cluster.
5. Repeat steps 3–4 until centroids no longer change (convergence).

### Example
If K=3 and you feed customer purchase data, K-Means may group customers into 3 clusters like "low spenders", "medium spenders" and "high spenders."

## Similar Question: What is the concept of convergence in K-means?

### Answer
Convergence occurs when centroids stabilize and data point assignments no longer change. Conditions for Convergence:
- Proper initialization (e.g., k-means++)
- Data naturally forms well-separated clusters
- Correct choice of number of clusters (k)
- Setting maximum iterations and tolerance for centroid changes

## Similar Question: How to choose an optimal number of clusters?

### Answer
- Elbow Method: Plot the explained variance or within-cluster sum of squares (WCSS) against the number of clusters. The "elbow" point where the curve starts to flatten, indicates the optimal number of clusters.
- Silhouette Score: Measures how similar each point is to its own cluster compared to other clusters. A higher silhouette score indicates better-defined clusters. The optimal number of clusters is the one with the highest average silhouette score.
- Gap Statistic: Compares the clustering result with a random clustering of the same data. A larger gap between the real and random clustering suggests a more appropriate number of clusters.

## Similar Question: What is the advanced version of K-Means?

### Answer
While K-Means is simple and widely used, it has limitations like sensitivity to outliers, need to predefine K and difficulty handling non-spherical clusters. Several advanced versions and alternatives improve upon it:
1. K-Medoids (PAM – Partitioning Around Medoids): Instead of using the mean as the cluster center, it uses an actual data point (medoid). More robust to outliers compared to K-Means.
2. K-Means++: An improved version of K-Means initialization. Chooses initial centroids more carefully which leads to better convergence and avoids poor clustering results.
3. Mini-Batch K-Means: Uses small random samples (mini-batches) of data instead of the entire dataset. Much faster and scalable for large datasets.
4. Fuzzy C-Means (Soft K-Means): Instead of hard assignment, each point has a probability of belonging to multiple clusters. Useful when clusters overlap.

## Similar Question: Explain K-Means++ and Fuzzy C-Means

### Answer
**K-Means++**
- K-Means++ is an improved version of the standard K-Means algorithm.
- The main improvement is in centroid initialization i.e instead of picking centroids randomly, it chooses them far apart from each other.
- This reduces the chances of poor clustering and improves convergence speed.
- Works the same as K-Means after initialization: assigns points to the nearest centroid and updates centroids iteratively.
- Best for: General clustering problems where standard K-Means may converge to a local minimum.

**Fuzzy C-Means (FCM)**
- Fuzzy C-Means is a soft clustering algorithm, unlike K-Means which assigns points to one cluster only.
- Each data point has a membership probability for each cluster, ranging between 0 and 1.
- Centroids are computed based on weighted membership values of points.
- Useful when clusters overlap or when strict cluster boundaries are unrealistic.
- Best for: Image segmentation, medical data or any scenario where a point can belong partially to multiple clusters.

---

# Clustering Evaluation Metrics

## Primary Question: You are working on a clustering problem, what are different evaluation metrics that can be used, and how to choose between them?

### Answer
Clusters are evaluated based on some similarity or dissimilarity measure such as the distance between cluster points. If the clustering algorithm separates dissimilar observations and similar observations together, then it has performed well. The two most popular metrics evaluation metrics for clustering algorithms are the Silhouette coefficient and Dunn's Index.

**Silhouette coefficient**
The Silhouette Coefficient is defined for each sample and is composed of two scores:
- a: The mean distance between a sample and all other points in the same cluster.
- b: The mean distance between a sample and all other points in the next nearest cluster.

The Silhouette coefficient for a set of samples is given as the mean of the Silhouette Coefficient for each sample. The score is bounded between -1 for incorrect clustering and +1 for highly dense clustering. Scores around zero indicate overlapping clusters. The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.

**Dunn's Index**
Dunn's Index (DI) is another metric for evaluating a clustering algorithm. Dunn's Index is equal to the minimum inter-cluster distance divided by the maximum cluster size. Note that large inter-cluster distances (better separation) and smaller cluster sizes (more compact clusters) lead to a higher DI value. A higher DI implies better clustering. It assumes that better clustering means that clusters are compact and well-separated from other clusters.

### Formula
S = (b − a) / max(a, b)

### Diagram
![Alt_text](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Figures/Derivation-of-the-Overall-Silhouette-Coefficient-OverallSil.png)

---

# Clustering Algorithms for Large Datasets

## Primary Question: Discuss two clustering algorithms that can scale to large datasets

### Answer
**Minibatch Kmeans:** Instead of using the full dataset at each iteration, the algorithm is capable of using mini-batches, moving the centroids just slightly at each iteration. This speeds up the algorithm typically by a factor of 3 or 4 and makes it possible to cluster huge datasets that do not fit in memory. Scikit-Learn implements this algorithm in the MiniBatchKMeans class.

**Balanced Iterative Reducing and Clustering using Hierarchies (BIRCH)** is a clustering algorithm that can cluster large datasets by first generating a small and compact summary of the large dataset that retains as much information as possible. This smaller summary is then clustered instead of clustering the larger dataset.

---

# Parametric vs Non-Parametric Models

## Primary Question: Define and compare parametric and non-parametric models and give two examples for each of them?

### Answer
**Parametric models** assume that the dataset comes from a certain function with some set of parameters that should be tuned to reach the optimal performance. For such models, the number of parameters is determined prior to training, thus the degree of freedom is limited and reduces the chances of overfitting.

Ex. Linear Regression, Logistic Regression, LDA

**Nonparametric models** don't assume anything about the function from which the dataset was sampled. For these models, the number of parameters is not determined prior to training, thus they are free to generalize the model based on the data. Sometimes these models overfit themselves while generalizing. To generalize they need more data in comparison with Parametric Models. They are relatively more difficult to interpret compared to Parametric Models.

Ex. Decision Tree, Random Forest.

---

# Squared Error vs Absolute Error

## Primary Question: What are the differences between a model that minimizes squared error and the one that minimizes the absolute error? and in which cases each error metric would be more appropriate?

### Answer
Both mean square error (MSE) and mean absolute error (MAE) measures the distances between vectors and express average model prediction in units of the target variable. Both can range from 0 to infinity, the lower they are the better the model.

The main difference between them is that in MSE the errors are squared before being averaged while in MAE they are not. This means that a large weight will be given to large errors. MSE is useful when large errors in the model are trying to be avoided. This means that outliers affect MSE more than MAE, that is why MAE is more robust to outliers.

Computation-wise MSE is easier to use as the gradient calculation will be more straightforward than MAE, which requires linear programming to calculate it.

---

# Instance-Based vs Model-Based Learning

## Primary Question: Instance-Based Versus Model-Based Learning.

### Answer
**Instance-based Learning:** The system learns the examples by heart, then generalizes to new cases using a similarity measure.

**Model-based Learning:** Another way to generalize from a set of examples is to build a model of these examples, then use that model to make predictions. This is called model-based learning.

---

# Supervised, Unsupervised, and Reinforcement Learning

## Primary Question: Can you explain the differences between supervised, unsupervised, and reinforcement learning?

### Answer
In supervised learning, we train a model to learn the relationship between input data and output data. We need to have labeled data to be able to do supervised learning.

With unsupervised learning, we only have unlabeled data. The model learns a representation of the data. Unsupervised learning is frequently used to initialize the parameters of the model when we have a lot of unlabeled data and a small fraction of labeled data. We first train an unsupervised model and, after that, we use the weights of the model to train a supervised model.

In reinforcement learning, the model has some input data and a reward depending on the output of the model. The model learns a policy that maximizes the reward. Reinforcement learning has been applied successfully to strategic games such as Go and even classic Atari video games.

---

# Data Augmentation

## Primary Question: What is data augmentation? Can you give some examples?

### Answer
Data augmentation is a technique for synthesizing new data by modifying existing data in such a way that the target is not changed, or it is changed in a known way.

Computer vision is one of the fields where data augmentation is very useful. There are many modifications that we can do to images:
- Resize
- Horizontal or vertical flip
- Rotate
- Add noise
- Deform
- Modify colors

Each problem needs a customized data augmentation pipeline. For example, on OCR, doing flips will change the text and won't be beneficial; however, resizes and small rotations may help.

---

# Turing Test

## Primary Question: What is Turing test?

### Answer
The Turing test is a method to test the machine's ability to match the human level intelligence. A machine is used to challenge the human intelligence that when it passes the test, it is considered as intelligent. Yet a machine could be viewed as intelligent without sufficiently knowing about people to mimic a human.

---

# Learning Rate

## Primary Question: Define Learning Rate.

### Answer
Learning rate is a hyper-parameter that controls how much we are adjusting the weights of our network with respect to the loss gradient.

---

# Momentum

## Primary Question: What is Momentum (w.r.t NN optimization)?

### Answer
Momentum lets the optimization algorithm remember its last step, and adds some proportion of it to the current step. This way, even if the algorithm is stuck in a flat region, or a small local minimum, it can get out and continue towards the true minimum.

---

# Epoch vs Batch vs Iteration

## Primary Question: Epoch vs. Batch vs. Iteration.

### Answer
- Epoch: one forward pass and one backward pass of all the training examples
- Batch: examples processed together in one pass (forward and backward)
- Iteration: number of training examples / Batch size

---

# Vanishing Gradient

## Primary Question: What is vanishing gradient?

### Answer
As we add more and more hidden layers, back propagation becomes less and less useful in passing information to the lower layers. In effect, as information is passed back, the gradients begin to vanish and become small relative to the weights of the networks.

---

# Dropout

## Primary Question: What are dropouts?

### Answer
Dropout is a simple way to prevent a neural network from overfitting. It is the dropping out of some of the units in a neural network. It is similar to the natural reproduction process, where nature produces offsprings by combining distinct genes (dropping out others) rather than strengthening the co-adapting of them.

---

# LSTM

## Primary Question: Define LSTM.

### Answer
Long Short Term Memory – are explicitly designed to address the long term dependency problem, by maintaining a state of what to remember and what to forget.

## Similar Question: List the key components of LSTM.

### Answer
- Gates (forget, Memory, update & Read)
- tanh(x) (values between -1 to 1)
- Sigmoid(x) (values between 0 to 1)

---

# LSTM vs Transformers

## Primary Question: What is the basic difference between LSTM and Transformers?

### Answer
LSTMs (Long Short Term Memory) models consist of RNN cells designed to store and manipulate information across time steps more efficiently. In contrast, Transformer models contain a stack of encoder and decoder layers, each consisting of self attention and feed-forward neural network components.

---

# RNN Variants

## Primary Question: List the variants of RNN.

### Answer
- LSTM: Long Short Term Memory
- GRU: Gated Recurrent Unit
- End to End Network
- Memory Network

---

# Autoencoder

## Primary Question: What is Autoencoder, name few applications.

### Answer
Auto encoder is basically used to learn a compressed form of given data. Few applications include:
- Data denoising
- Dimensionality reduction
- Image reconstruction
- Image colorization

---

# GAN

## Primary Question: What are the components of GAN?

### Answer
- Generator
- Discriminator

---

# Batch Normalization

## Primary Question: What is batch normalization and why does it work?

### Answer
Training Deep Neural Networks is complicated by the fact that the distribution of each layer's inputs changes during training, as the parameters of the previous layers change. The idea is then to normalize the inputs of each layer in such a way that they have a mean output activation of zero and standard deviation of one. This is done for each individual mini-batch at each layer i.e compute the mean and variance of that mini-batch alone, then normalize. This is analogous to how the inputs to networks are standardized.

How does this help? We know that normalizing the inputs to a network helps it learn. But a network is just a series of layers, where the output of one layer becomes the input to the next. That means we can think of any layer in a neural network as the first layer of a smaller subsequent network. Thought of as a series of neural networks feeding into each other, we normalize the output of one layer before applying the activation function, and then feed it into the following layer (sub-network).

---

# Residual Networks

## Primary Question: What is the significance of Residual Networks?

### Answer
The main thing that residual connections did was allow for direct feature access from previous layers. This makes information propagation throughout the network much easier. One very interesting finding about this shows how using local skip connections gives the network a type of ensemble multi-path structure, giving features multiple paths to propagate throughout the network.

---

# Convolution - How It Works

## Primary Question: Describe how convolution works. What about if your inputs are grayscale vs. RGB imagery? What determines the shape of the next layer?

### Answer
In a convolutional neural network (CNN), the convolution operation is applied to the input image using a small matrix called a kernel or filter. The kernel slides over the image in small steps, called strides, and performs element-wise multiplications with the corresponding elements of the image and then sums up the results. The output of this operation is called a feature map.

When the input is RGB(or more than 3 channels) the sliding window will be a sliding cube. The shape of the next layer is determined by Kernel size, number of kernels, stride, padding, and dilation.

---

# Why Convolutions Instead of Fully Connected Layers

## Primary Question: Why do we use convolutions for images rather than just FC layers?

### Answer
Firstly, convolutions preserve, encode, and actually use the spatial information from the image. If we used only FC layers we would have no relative spatial information. Secondly, Convolutional Neural Networks (CNNs) have a partially built-in translation in-variance, since each convolution kernel acts as its own filter/feature detector.

---

# CNN Translation Invariance

## Primary Question: What makes CNNs translation invariant?

### Answer
As explained above, each convolution kernel acts as its own filter/feature detector. So let's say you're doing object detection, it doesn't matter where in the image the object is since we're going to apply the convolution in a sliding window fashion across the entire image anyways.

---

# Max Pooling in CNNs

## Primary Question: Why do we have max-pooling in classification CNNs?

### Answer
Max-pooling in a CNN allows you to reduce computation since your feature maps are smaller after the pooling. You don't lose too much semantic information since you're taking the maximum activation. There's also a theory that max-pooling contributes a bit to giving CNNs more translation in-variance.

---

# Encoder-Decoder Structure in Segmentation CNNs

## Primary Question: Why do segmentation CNNs typically have an encoder-decoder style / structure?

### Answer
The encoder CNN can basically be thought of as a feature extraction network, while the decoder uses that information to predict the image segments by "decoding" the features and upscaling to the original image size.

---

# Small vs Large Convolutional Kernels

## Primary Question: Why would you use many small convolutional kernels such as 3x3 rather than a few large ones?

### Answer
This is very well explained in the VGGNet paper. There are 2 reasons: First, you can use several smaller kernels rather than few large ones to get the same receptive field and capture more spatial context, but with the smaller kernels you are using less parameters and computations. Secondly, because with smaller kernels you will be using more filters, you'll be able to use more activation functions and thus have a more discriminative mapping function being learned by your CNN.

---

# CNN Receptive Field Calculation

## Primary Question: Given stride S and kernel sizes for each layer of a (1-dimensional) CNN, create a function to compute the receptive field of a particular node in the network. This is just finding how many input nodes actually connect through to a neuron in a CNN.

### Answer
The receptive field is defined as the portion of space within the inputs that will be used during an operation to generate an output.

Considering a CNN filter of size k, the receptive field of a particular layer is only the number of input used by the filter, in this case k, multiplied by the dimension of the input that is not being reduced by the convolutional filter a. This results in a receptive field of k*a.

More visually, in the case of an image of size 32x32x3, with a CNN with a filter size of 5x5, the corresponding receptive field will be the filter size, 5 multiplied by the depth of the input volume (the RGB colors) which is the color dimension. This thus gives us a receptive field of dimension 5x5x3.

---

# Data Normalization

## Primary Question: What is data normalization and why do we need it?

### Answer
Data normalization is a very important preprocessing step, used to rescale values to fit in a specific range to assure better convergence during backpropagation. In general, it boils down to subtracting the mean of each data point and dividing by its standard deviation. If we don't do this then some of the features (those with high magnitude) will be weighted more in the cost function (if a higher-magnitude feature changes by 1%, then that change is pretty big, but for smaller features it's quite insignificant). The data normalization makes all features weighted equally.

### Quick Explanation (AI Generated)
Regularization, Standardization and Normalization are related but distinct concepts from the notes:
- Regularization: prevents overfitting by penalizing large weights in the loss function (e.g. L1/L2).
- Standardization: rescales features to mean 0, standard deviation 1; useful for Logistic Regression, SVM, K-Means, PCA.
- Normalization: rescales features to a fixed range (usually 0 to 1); useful in Neural Networks, KNN, image processing.

---

# Validation Set vs Test Set

## Primary Question: Why do we need a validation set and test set? What is the difference between them?

### Answer
When training a model, we divide the available data into three separate sets:

The training dataset is used for fitting the model's parameters. However, the accuracy that we achieve on the training set is not reliable for predicting if the model will be accurate on new samples.

The validation dataset is used to measure how well the model does on examples that weren't part of the training dataset. The metrics computed on the validation data can be used to tune the hyperparameters of the model. However, every time we evaluate the validation data and we make decisions based on those scores, we are leaking information from the validation data into our model. The more evaluations, the more information is leaked. So we can end up overfitting to the validation data, and once again the validation score won't be reliable for predicting the behaviour of the model in the real world.

The test dataset is used to measure how well the model does on previously unseen examples. It should only be used once we have tuned the parameters using the validation set.

So if we omit the test set and only use a validation set, the validation score won't be a good estimate of the generalization of the model.

---

# Model Evaluation Splitting Mechanism (Multiple Models)

## Primary Question: Given that we want to evaluate the performance of 'n' different machine learning models on the same data, why would the following splitting mechanism be incorrect:

### Code Example
```python
def get_splits():
    df = pd.DataFrame(...)
    rnd = np.random.rand(len(df))
    train = df[ rnd < 0.8 ]
    valid = df[ rnd >= 0.8 & rnd < 0.9 ]
    test = df[ rnd >= 0.9 ]

    return train, valid, test

#Model 1

from sklearn.tree import DecisionTreeClassifier
train, valid, test = get_splits()
...

#Model 2

from sklearn.linear_model import LogisticRegression
train, valid, test = get_splits()
...
```

### Answer
The rand() function orders the data differently each time it is run, so if we run the splitting mechanism again, the 80% of the rows we get will be different from the ones we got the first time it was run. This presents an issue as we need to compare the performance of our models on the same test set. In order to ensure reproducible and consistent sampling we would have to set the random seed in advance or store the data once it is split. Alternatively, we could simply set the 'random_state' parameter in sklearn's train_test_split() function in order to get the same train, validation and test sets across different executions.

---

# Active Learning

## Primary Question: What is active learning and discuss one strategy of it?

### Answer
Active learning is a special case of machine learning in which a learning algorithm can interactively query a user (or some other information source) to label new data points with the desired outputs. In statistics literature, it is sometimes referred to as optimal experimental design.

1. Stream-based sampling: In stream-based selective sampling, unlabelled data is continuously fed to an active learning system, where the learner decides whether to send the same to a human oracle or not based on a predefined learning strategy. This method is apt in scenarios where the model is in production and the data sources/distributions vary over time.

2. Pool-based sampling: In this case, the data samples are chosen from a pool of unlabelled data based on the informative value scores and sent for manual labeling. Unlike stream-based sampling, oftentimes, the entire unlabelled dataset is scrutinized for the selection of the best instances.

### Diagram
![1669836673164](https://user-images.githubusercontent.com/72076328/204896144-43b2181a-d9ce-471b-95d0-44c0f7bb3025.jpg)

---

# Recommendation Systems

## Primary Question: What are the different approaches to implementing recommendation systems?

### Answer
1. **Content-Based Filtering:** Content-Based Filtering depends on similarities of items and users' past activities on the website to recommend any product or service. This filter helps in avoiding a cold start for any new products as it doesn't rely on other users' feedback, it can recommend products based on similarity factors. However, content-based filtering needs a lot of domain knowledge so that the recommendations made are 100 percent accurate.

It recommends items based on the item's own features (genre, keywords, specifications) matched against what the user has liked before. Does not need data from other users. Handles the new item cold-start problem reasonably well. Struggles with the new user cold-start problem.

Working: Represent items using a feature vector. Compare the feature vectors of new items with items the user liked using similarity metrics (e.g., cosine similarity). Recommend items with the highest similarity scores.

2. **Collaborative-Based Filtering:** The primary job of a collaborative filtering system is to overcome the shortcomings of content-based filtering. So, instead of focusing on just one user, the collaborative filtering system focuses on all the users and clusters them according to their interests. Basically, it recommends a product 'x' to user 'a' based on the interest of user 'b'; users 'a' and 'b' must have had similar interests in the past, which is why they are clustered together.

The domain knowledge that is required for collaborative filtering is less, recommendations made are more accurate and it can adapt to the changing tastes of users over time. However, collaborative filtering faces the problem of a cold start as it heavily relies on feedback or activity from other users.

Working: User-Based CF: Finds users similar to the target user and recommends items they liked. Item-Based CF: Finds items similar to those the user liked and recommends them.

3. **Hybrid filtering:** A mixture of content and collaborative methods. Uses descriptors and interactions. More modern approaches typically fall into the hybrid filtering category and tend to work in two stages:
   1. A candidate generation phase where we coarsely generate candidates from a corpus of hundreds of thousands, millions, or billions of items down to a few hundred or thousand.
   2. A ranking phase where we re-rank the candidates into a final top-n set to be shown to the user. Some systems employ multiple candidate generation methods and rankers.

### Example
A user watches movies with action and sci-fi genres. The system recommends other action or sci-fi movies based on these features (content-based). User A and user B both liked movies X and Y. User A liked movie Z, so movie Z is recommended to User B (collaborative).

---

# Multi-Label Classification Evaluation Metrics

## Primary Question: What are the evaluation metrics that can be used for multi-label classification?

### Answer
Multi-label classification is a type of classification problem where each instance can be assigned to multiple classes or labels simultaneously.

The evaluation metrics for multi-label classification are designed to measure the performance of a multi-label classifier in predicting the correct set of labels for each instance. Some commonly used evaluation metrics for multi-label classification are:

1. Hamming Loss: Hamming Loss is the fraction of labels that are incorrectly predicted. It is defined as the average number of labels that are predicted incorrectly per instance.
2. Accuracy: Accuracy is the fraction of instances that are correctly predicted. In multi-label classification, accuracy is calculated as the percentage of instances for which all labels are predicted correctly.
3. Precision, Recall, F1-Score: These metrics can be applied to each label separately, treating the classification of each label as a separate binary classification problem. Precision measures the proportion of predicted positive labels that are correct, recall measures the proportion of actual positive labels that are correctly predicted, and F1-score is the harmonic mean of precision and recall.
4. Macro-F1, Micro-F1: Macro-F1 and Micro-F1 are two types of F1-score metrics that take into account the label imbalance in the dataset. Macro-F1 calculates the F1-score for each label and then averages them, while Micro-F1 calculates the overall F1-score by aggregating the true positive, false positive, and false negative counts across all labels.

There are other metrics that can be used such as:
- Precision at k (P@k)
- Average precision at k (AP@k)
- Mean average precision at k (MAP@k)

---

# Concept Drift vs Data Drift

## Primary Question: What is the difference between concept and data drift and how to overcome each of them?

### Answer
Concept drift and data drift are two different types of problems that can occur in machine learning systems.

Concept drift refers to changes in the underlying relationships between the input data and the target variable over time. This means that the distribution of the data that the model was trained on no longer matches the distribution of the data it is being tested on. For example, a spam filter model that was trained on emails from several years ago may not be as effective at identifying spam emails from today because the language and tactics used in spam emails may have changed.

Data drift, on the other hand, refers to changes in the input data itself over time. This means that the values of the input feature that the model was trained on no longer match the values of the input features in the data it is being tested on. For example, a model that was trained on data from a particular geographical region may not be as effective at predicting outcomes for data from a different region.

To overcome concept drift, one approach is to use online learning methods that allow the model to adapt to new data as it arrives. This involves continually training the model on the most recent data while using historical data to maintain context. Another approach is to periodically retrain the model using a representative sample of the most recent data.

To overcome data drift, one approach is to monitor the input data for changes and retrain the model when significant changes are detected. This may involve setting up a monitoring system that alerts the user when the data distribution changes beyond a certain threshold. Another approach is to preprocess the input data to remove or mitigate the effects of the features changing over time so that the model can continue learning from the remaining features.

Types of Concept Drift:
- Sudden Drift: The data distribution changes abruptly.
- Gradual Drift: The change happens slowly over time.
- Recurring Drift: Old patterns reappear after some time.

Handling Concept Drift: Regularly retraining the model with new data. Using adaptive learning algorithms that update as new data arrives.

### Diagram
![ezgif com-webp-to-jpg (7)](https://user-images.githubusercontent.com/72076328/221916192-7a9fcf21-8e5f-4ddc-bd90-ef1bdabf1d3f.jpg)

---

# ARIMA Model

## Primary Question: Can you explain the ARIMA model and its components?

### Answer
The ARIMA model, which stands for Autoregressive Integrated Moving Average, is a widely used time series forecasting model. It combines three key components: Autoregression (AR), Differencing (I), and Moving Average (MA).

- **Autoregression (AR):** The autoregressive component captures the relationship between an observation in a time series and a certain number of lagged observations. It assumes that the value at a given time depends linearly on its own previous values. The "p" parameter in ARIMA(p, d, q) represents the order of autoregressive terms. For example, ARIMA(1, 0, 0) refers to a model with one autoregressive term.

- **Differencing (I):** Differencing is used to make a time series stationary by removing trends or seasonality. It calculates the difference between consecutive observations to eliminate any non-stationary behavior. The "d" parameter in ARIMA(p, d, q) represents the order of differencing. For instance, ARIMA(0, 1, 0) indicates that differencing is applied once.

- **Moving Average (MA):** The moving average component takes into account the dependency between an observation and a residual error from a moving average model applied to lagged observations. It assumes that the value at a given time depends linearly on the error terms from previous time steps. The "q" parameter in ARIMA(p, d, q) represents the order of the moving average terms. For example, ARIMA(0, 0, 1) signifies a model with one moving average term.

By combining these three components, the ARIMA model can capture both autoregressive patterns, temporal dependencies, and stationary behavior in a time series. The parameters p, d, and q are typically determined through techniques like the Akaike Information Criterion (AIC) or Bayesian Information Criterion (BIC).

It's worth noting that there are variations of the ARIMA model, such as SARIMA (Seasonal ARIMA), which incorporates additional seasonal components for modeling seasonal patterns in the data.

ARIMA models are widely used in forecasting applications, but they do make certain assumptions about the underlying data, such as linearity and stationarity. It's important to validate these assumptions and adjust the model accordingly if they are not met.

### Diagram
![1-1](https://github.com/youssefHosni/Data-Science-Interview-Questions-Answers/assets/72076328/12707951-bdf5-4cd1-9efd-c60c465007a3)

## Similar Question: What are the assumptions made by the ARIMA model?

### Answer
The ARIMA model makes several assumptions about the underlying time series data. These assumptions are important to ensure the validity and accuracy of the model's results. Here are the key assumptions:

**Stationarity:** The ARIMA model assumes that the time series is stationary. Stationarity means that the statistical properties of the data, such as the mean and variance, remain constant over time. This assumption is crucial for the autoregressive and moving average components to hold. If the time series is non-stationary, differencing (the "I" component) is applied to transform it into a stationary series.

**Linearity:** The ARIMA model assumes that the relationship between the observations and the lagged values is linear. It assumes that the future values of the time series can be modeled as a linear combination of past values and error terms.

**No Autocorrelation in Residuals:** The ARIMA model assumes that the residuals (the differences between the predicted values and the actual values) do not exhibit any autocorrelation. In other words, the errors are not correlated with each other.

**Normally Distributed Residuals:** The ARIMA model assumes that the residuals follow a normal distribution with a mean of zero. This assumption is necessary for statistical inference, parameter estimation, and hypothesis testing.

It's important to note that while these assumptions are commonly made in ARIMA modeling, they may not always hold in real-world scenarios. It's essential to assess the data and, if needed, apply transformations or consider alternative models that relax some of these assumptions. Additionally, diagnostics tools, such as residual analysis and statistical tests, can help evaluate the adequacy of the assumptions and the model's fit to the data.

---

# Bayesian vs Frequentist Statistics

## Primary Question: What is the difference between Bayesian vs frequentist statistics?

### Answer
Frequentist statistics is a framework that focuses on estimating population parameters using sample statistics, and providing point estimates and confidence intervals.

Bayesian statistics, on the other hand, is a framework that uses prior knowledge and information to update beliefs about a parameter or hypothesis, and provides probability distributions for parameters.

The main difference is that Bayesian statistics incorporates prior knowledge and beliefs into the analysis, while frequentist statistics doesn't.

---

# Random Number Generator

## Primary Question: How Random Number Generator Works, e.g. rand() function in python works?

### Answer
It generates a pseudo random number based on the seed and there are some famous algorithms for it.

---

# RCNN

## Primary Question: What are RCNNs?

### Answer
Recurrent Convolutional model is a model that is specially designed to make predictions using a sequence of images (more commonly also known as video). These models are used in object detection tasks in computer vision. The RCNN approach combines both region proposal techniques and convolutional neural networks (CNNs) to identify and locate objects within an image.

---

# CBIR (Content-Based Image Retrieval)

## Primary Question: How does CBIR work?

### Answer
Content-based image retrieval is the concept of using images to gather metadata on their content. Compared to the current image retrieval approach based on the keywords associated to the images, this technique generates its metadata from computer vision techniques to extract the relevant information that will be used during the querying step. Many approaches are possible from feature detection to retrieve keywords to the usage of CNN to extract dense features that will be associated to a known distribution of keywords.

With this last approach, we care less about what is shown on the image but more about the similarity between the metadata generated by a known image and a list of known label and or tags projected into this metadata space.

---

# Image Registration and Optical Flow

## Primary Question: How does image registration work? Sparse vs. dense optical flow and so on.

### Answer
> Answer not available in the provided notes.

---

# 3D Model Reconstruction from Imagery/Depth

## Primary Question: Talk me through how you would create a 3D model of an object from imagery and depth sensor measurements taken at all angles around the object.

### Answer
There are two popular methods for 3D reconstruction:

Structure from Motion (SfM)

Multi-View Stereo (MVS)

SfM is better suited for creating models of large scenes while MVS is better suited for creating models of small objects.

---

# Outlier Removal in Plane Estimation (RANSAC-style)

## Primary Question: How would you remove outliers when trying to estimate a flat plane from noisy samples?

### Answer
Random sample consensus (RANSAC) is an iterative method to estimate parameters of a mathematical model from a set of observed data that contains outliers, when outliers are to be accorded no influence on the values of the estimates.

---

# Connected Components Implementation

## Primary Question: Implement connected components on an image/matrix.

### Answer
> Answer not available in the provided notes.

---

# Sparse Matrix Class Implementation

## Primary Question: Implement a sparse matrix class in C++.

### Answer
> Answer not available in the provided notes.

---

# Integral Image Computation

## Primary Question: Create a function to compute an integral image, and create another function to get area sums from the integral image.

### Answer
> Answer not available in the provided notes.

---

# SQRT Implementation

## Primary Question: Implement SQRT(const double & x) without using any special functions, just fundamental arithmetic.

### Answer
The taylor series can be used for this step by providing an approximation of sqrt(x).

> Note: The notes reference this approach but do not provide the actual implementation content.

---

# Reverse a Bitstring

## Primary Question: Reverse a bitstring.

### Answer
If you are using python3:

### Code Example
```python
data = b'\xAD\xDE\xDE\xC0'
my_data = bytearray(data)
my_data.reverse()
```

---

# Non-Maximal Suppression Implementation

## Primary Question: Implement non maximal suppression as efficiently as you can.

### Answer
Non-Maximum Suppression (NMS) is a technique used to eliminate multiple detections of the same object in a given image. To solve that first sort bounding boxes based on their scores(N LogN). Starting with the box with the highest score, remove boxes whose overlapping metric(IoU) is greater than a certain threshold.(N^2)

To optimize this solution you can use special data structures to query for overlapping boxes such as R-tree or KD-tree. (N LogN)

---

# Reverse a Linked List

## Primary Question: Reverse a linked list in place.

### Answer
> Answer not available in the provided notes.
