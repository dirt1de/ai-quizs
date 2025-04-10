import { Question } from '../types/quiz';

export const questions: Question[] = [
  {
    id: 1,
    question: "What is the main difference between L1 (Lasso) and L2 (Ridge) regularization?",
    options: [
      "L1 uses absolute values, L2 uses squared values",
      "L1 promotes sparsity while L2 doesn't",
      "L1 is computationally faster than L2",
      "L1 and L2 are the same thing"
    ],
    correctAnswer: 1,
    explanation: "L1 (Lasso) regularization promotes sparsity by potentially setting some coefficients to exactly zero, while L2 (Ridge) shrinks coefficients close to zero but rarely exactly zero.",
    reference: "https://scikit-learn.org/stable/modules/linear_model.html#lasso"
  },
  {
    id: 2,
    question: "What is the purpose of Principal Component Analysis (PCA)?",
    options: [
      "To classify data points",
      "To reduce dimensionality while preserving maximum variance",
      "To increase the number of features",
      "To normalize the data"
    ],
    correctAnswer: 1,
    explanation: "PCA is a dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space while preserving as much variance as possible.",
    reference: "https://scikit-learn.org/stable/modules/decomposition.html#pca"
  },
  {
    id: 3,
    question: "What is the difference between supervised and unsupervised learning?",
    options: [
      "Supervised learning uses labeled data, unsupervised learning doesn't",
      "Supervised learning is faster than unsupervised learning",
      "Supervised learning always gives better results",
      "There is no difference"
    ],
    correctAnswer: 0,
    explanation: "Supervised learning uses labeled data to train models, while unsupervised learning finds patterns in unlabeled data without predefined outputs.",
    reference: "https://towardsdatascience.com/supervised-vs-unsupervised-learning-14f68e32ea8d"
  },
  {
    id: 4,
    question: "What is the purpose of the confusion matrix?",
    options: [
      "To visualize model performance metrics",
      "To store training data",
      "To optimize model parameters",
      "To preprocess data"
    ],
    correctAnswer: 0,
    explanation: "A confusion matrix is a table that visualizes the performance of a classification model by showing true positives, true negatives, false positives, and false negatives.",
    reference: "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html"
  },
  {
    id: 5,
    question: "What is the difference between precision and recall?",
    options: [
      "Precision measures true positives among predicted positives, recall measures true positives among actual positives",
      "Precision and recall are the same thing",
      "Precision is always higher than recall",
      "Recall is more important than precision"
    ],
    correctAnswer: 0,
    explanation: "Precision = TP/(TP+FP), while Recall = TP/(TP+FN). Precision focuses on the accuracy of positive predictions, while recall focuses on finding all positive cases.",
    reference: "https://towardsdatascience.com/precision-vs-recall-386cf9f89488"
  },
  {
    id: 6,
    question: "What is the purpose of feature scaling?",
    options: [
      "To make all features have the same scale",
      "To remove outliers",
      "To increase model complexity",
      "To reduce the number of features"
    ],
    correctAnswer: 0,
    explanation: "Feature scaling standardizes the range of independent variables, which is important for many machine learning algorithms that are sensitive to the scale of features.",
    reference: "https://scikit-learn.org/stable/modules/preprocessing.html#scaling-features-to-a-range"
  },
  {
    id: 7,
    question: "What is the difference between bagging and boosting?",
    options: [
      "Bagging builds models in parallel, boosting builds models sequentially",
      "Bagging is faster than boosting",
      "Bagging always gives better results",
      "There is no difference"
    ],
    correctAnswer: 0,
    explanation: "Bagging builds multiple models in parallel and averages their predictions, while boosting builds models sequentially, with each new model focusing on the errors of previous models.",
    reference: "https://towardsdatascience.com/ensemble-methods-bagging-boosting-and-stacking-c9214a10a205"
  },
  {
    id: 8,
    question: "What is the purpose of the ROC curve?",
    options: [
      "To visualize the trade-off between true positive rate and false positive rate",
      "To store training data",
      "To optimize model parameters",
      "To preprocess data"
    ],
    correctAnswer: 0,
    explanation: "The ROC curve plots the true positive rate against the false positive rate at various threshold settings, helping to evaluate the performance of a binary classifier.",
    reference: "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html"
  },
  {
    id: 9,
    question: "What is the difference between correlation and causation?",
    options: [
      "Correlation doesn't imply causation",
      "Correlation always implies causation",
      "Causation is easier to prove than correlation",
      "They are the same thing"
    ],
    correctAnswer: 0,
    explanation: "Correlation indicates a relationship between variables, but doesn't prove that one causes the other. Causation requires additional evidence beyond correlation.",
    reference: "https://www.statisticshowto.com/correlation-vs-causation/"
  },
  {
    id: 10,
    question: "What is the purpose of cross-validation?",
    options: [
      "To assess model performance on unseen data",
      "To increase model complexity",
      "To reduce the number of features",
      "To make the model run faster"
    ],
    correctAnswer: 0,
    explanation: "Cross-validation helps evaluate how a model will generalize to independent data and prevents overfitting by using multiple train-test splits.",
    reference: "https://scikit-learn.org/stable/modules/cross_validation.html"
  },
  {
    id: 11,
    question: "What is the difference between overfitting and underfitting?",
    options: [
      "Overfitting captures noise, underfitting misses patterns",
      "Overfitting is always better than underfitting",
      "Underfitting is always better than overfitting",
      "They are the same thing"
    ],
    correctAnswer: 0,
    explanation: "Overfitting occurs when a model captures noise in the training data, while underfitting occurs when a model fails to capture the underlying patterns.",
    reference: "https://towardsdatascience.com/overfitting-vs-underfitting-a-complete-example-d05dd7e19765"
  },
  {
    id: 12,
    question: "What is the purpose of feature selection?",
    options: [
      "To choose the most relevant features",
      "To increase model complexity",
      "To make the model run faster",
      "To remove all features"
    ],
    correctAnswer: 0,
    explanation: "Feature selection helps identify and select the most relevant features for model building, which can improve model performance and reduce overfitting.",
    reference: "https://scikit-learn.org/stable/modules/feature_selection.html"
  },
  {
    id: 13,
    question: "What is the difference between classification and regression?",
    options: [
      "Classification predicts categories, regression predicts continuous values",
      "Classification is always better than regression",
      "Regression is always better than classification",
      "They are the same thing"
    ],
    correctAnswer: 0,
    explanation: "Classification predicts discrete categories, while regression predicts continuous numerical values.",
    reference: "https://machinelearningmastery.com/classification-versus-regression-in-machine-learning/"
  },
  {
    id: 14,
    question: "What is the purpose of the bias-variance tradeoff?",
    options: [
      "To balance model complexity and prediction error",
      "To make the model run faster",
      "To increase model complexity",
      "To reduce the number of features"
    ],
    correctAnswer: 0,
    explanation: "The bias-variance tradeoff helps find the right balance between model complexity and prediction error, avoiding both underfitting and overfitting.",
    reference: "https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229"
  },
  {
    id: 15,
    question: "What is the difference between batch gradient descent and stochastic gradient descent?",
    options: [
      "Batch uses all data, stochastic uses one sample at a time",
      "Batch is always faster than stochastic",
      "Stochastic is always better than batch",
      "They are the same thing"
    ],
    correctAnswer: 0,
    explanation: "Batch gradient descent updates parameters using the entire dataset, while stochastic gradient descent updates parameters using one sample at a time.",
    reference: "https://towardsdatascience.com/batch-mini-batch-stochastic-gradient-descent-7a62ecba642a"
  },
  {
    id: 16,
    question: "What is the purpose of a convolutional neural network (CNN)?",
    options: [
      "To process grid-like data such as images",
      "To handle sequential data",
      "To perform dimensionality reduction",
      "To classify text data"
    ],
    correctAnswer: 0,
    explanation: "CNNs are specifically designed to process grid-like data such as images, using convolutional layers to detect spatial patterns and hierarchies of features.",
    reference: "https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53"
  },
  {
    id: 17,
    question: "What is the purpose of dropout in neural networks?",
    options: [
      "To prevent overfitting by randomly deactivating neurons",
      "To increase model complexity",
      "To speed up training",
      "To reduce the number of parameters"
    ],
    correctAnswer: 0,
    explanation: "Dropout randomly deactivates neurons during training, which helps prevent overfitting by making the network more robust and less dependent on specific neurons.",
    reference: "https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/"
  },
  {
    id: 18,
    question: "What is the purpose of batch normalization?",
    options: [
      "To stabilize and accelerate training of deep neural networks",
      "To increase model complexity",
      "To reduce the number of parameters",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "Batch normalization normalizes the inputs of each layer, which helps stabilize and accelerate the training of deep neural networks.",
    reference: "https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c"
  },
  {
    id: 19,
    question: "What is the purpose of a recurrent neural network (RNN)?",
    options: [
      "To handle sequential data with temporal dependencies",
      "To process grid-like data",
      "To perform dimensionality reduction",
      "To classify images"
    ],
    correctAnswer: 0,
    explanation: "RNNs are designed to handle sequential data by maintaining a hidden state that captures information about previous elements in the sequence.",
    reference: "https://towardsdatascience.com/understanding-rnn-and-lstm-f7cdf6dfc14e"
  },
  {
    id: 20,
    question: "What is the purpose of an autoencoder?",
    options: [
      "To learn efficient data representations in an unsupervised manner",
      "To classify data points",
      "To predict continuous values",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "Autoencoders learn to compress and reconstruct data, creating efficient representations that capture the most important features of the input data.",
    reference: "https://towardsdatascience.com/auto-encoder-what-is-it-and-what-is-it-used-for-part-1-3e5c6f017726"
  },
  {
    id: 21,
    question: "What is the purpose of transfer learning?",
    options: [
      "To leverage knowledge from pre-trained models",
      "To increase model complexity",
      "To reduce training time",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "Transfer learning allows models to leverage knowledge gained from solving one problem to solve a different but related problem, often requiring less data and training time.",
    reference: "https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a"
  },
  {
    id: 22,
    question: "What is the purpose of data augmentation?",
    options: [
      "To increase the size and diversity of training data",
      "To reduce model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "Data augmentation creates new training examples by applying transformations to existing data, helping to prevent overfitting and improve model generalization.",
    reference: "https://towardsdatascience.com/data-augmentation-techniques-in-python-f216ef5eed69"
  },
  {
    id: 23,
    question: "What is the purpose of early stopping?",
    options: [
      "To prevent overfitting by stopping training when validation performance degrades",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "Early stopping monitors validation performance during training and stops when it starts to degrade, preventing overfitting and saving computational resources.",
    reference: "https://machinelearningmastery.com/early-stopping-to-avoid-overtraining-neural-network-models/"
  },
  {
    id: 24,
    question: "What is the purpose of a learning rate schedule?",
    options: [
      "To adaptively adjust the learning rate during training",
      "To increase model complexity",
      "To reduce the number of parameters",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "Learning rate schedules adjust the learning rate during training to help models converge faster and achieve better performance.",
    reference: "https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1"
  },
  {
    id: 25,
    question: "What is the purpose of gradient clipping?",
    options: [
      "To prevent exploding gradients in deep networks",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "Gradient clipping limits the size of gradients during backpropagation, preventing the exploding gradient problem in deep neural networks.",
    reference: "https://machinelearningmastery.com/how-to-avoid-exploding-gradients-in-neural-networks-with-gradient-clipping/"
  },
  {
    id: 26,
    question: "What is the purpose of weight initialization?",
    options: [
      "To set initial weights that help networks train effectively",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "Proper weight initialization helps networks start training from a good initial state, preventing issues like vanishing or exploding gradients.",
    reference: "https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79"
  },
  {
    id: 27,
    question: "What is the purpose of a validation set?",
    options: [
      "To tune hyperparameters and prevent overfitting",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "The validation set is used to tune hyperparameters and monitor model performance during training, helping to prevent overfitting to the training data.",
    reference: "https://machinelearningmastery.com/difference-test-validation-datasets/"
  },
  {
    id: 28,
    question: "What is the purpose of a test set?",
    options: [
      "To evaluate the final model performance",
      "To tune hyperparameters",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "The test set is used to evaluate the final model performance on unseen data, providing an unbiased estimate of how the model will perform in practice.",
    reference: "https://machinelearningmastery.com/difference-test-validation-datasets/"
  },
  {
    id: 29,
    question: "What is the purpose of k-fold cross-validation?",
    options: [
      "To get a more robust estimate of model performance",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "K-fold cross-validation provides a more robust estimate of model performance by averaging results across multiple train-test splits.",
    reference: "https://machinelearningmastery.com/k-fold-cross-validation/"
  },
  {
    id: 30,
    question: "What is the purpose of a confusion matrix?",
    options: [
      "To evaluate classification model performance",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "A confusion matrix provides a detailed breakdown of classification model performance, showing true positives, true negatives, false positives, and false negatives.",
    reference: "https://machinelearningmastery.com/confusion-matrix-machine-learning/"
  },
  {
    id: 31,
    question: "What is the purpose of precision and recall?",
    options: [
      "To evaluate classification model performance in imbalanced datasets",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "Precision and recall are particularly useful for evaluating classification models on imbalanced datasets, where accuracy can be misleading.",
    reference: "https://machinelearningmastery.com/precision-recall-and-f-measure-for-imbalanced-classification/"
  },
  {
    id: 32,
    question: "What is the purpose of the F1 score?",
    options: [
      "To balance precision and recall in a single metric",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "The F1 score combines precision and recall into a single metric, providing a balanced measure of model performance.",
    reference: "https://machinelearningmastery.com/precision-recall-and-f-measure-for-imbalanced-classification/"
  },
  {
    id: 33,
    question: "What is the purpose of the ROC curve?",
    options: [
      "To visualize the trade-off between true positive rate and false positive rate",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "The ROC curve helps visualize and evaluate the performance of a binary classifier across different classification thresholds.",
    reference: "https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/"
  },
  {
    id: 34,
    question: "What is the purpose of the AUC score?",
    options: [
      "To measure the overall performance of a binary classifier",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "The AUC score provides a single number that summarizes the overall performance of a binary classifier across all possible classification thresholds.",
    reference: "https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/"
  },
  {
    id: 35,
    question: "What is the purpose of feature engineering?",
    options: [
      "To create better input features for machine learning models",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "Feature engineering involves creating new features or transforming existing ones to improve model performance and better capture the underlying patterns in the data.",
    reference: "https://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/"
  },
  {
    id: 36,
    question: "What is the purpose of one-hot encoding?",
    options: [
      "To convert categorical variables into numerical format",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "One-hot encoding converts categorical variables into a binary format that can be used by machine learning algorithms.",
    reference: "https://machinelearningmastery.com/one-hot-encoding-for-categorical-data/"
  },
  {
    id: 37,
    question: "What is the purpose of label encoding?",
    options: [
      "To convert categorical variables into numerical format",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "Label encoding assigns a unique numerical value to each category, converting categorical variables into a format that can be used by machine learning algorithms.",
    reference: "https://machinelearningmastery.com/one-hot-encoding-for-categorical-data/"
  },
  {
    id: 38,
    question: "What is the purpose of standardization?",
    options: [
      "To scale features to have zero mean and unit variance",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "Standardization transforms features to have zero mean and unit variance, which is important for many machine learning algorithms.",
    reference: "https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/"
  },
  {
    id: 39,
    question: "What is the purpose of normalization?",
    options: [
      "To scale features to a specific range",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "Normalization scales features to a specific range, typically [0, 1], which can be important for certain machine learning algorithms.",
    reference: "https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/"
  },
  {
    id: 40,
    question: "What is the purpose of imputation?",
    options: [
      "To handle missing data",
      "To increase model complexity",
      "To speed up training",
      "To reduce the number of features"
    ],
    correctAnswer: 0,
    explanation: "Imputation fills in missing values in a dataset, allowing machine learning algorithms to work with incomplete data.",
    reference: "https://machinelearningmastery.com/handle-missing-data-python/"
  },
  {
    id: 41,
    question: "What is the purpose of outlier detection?",
    options: [
      "To identify and handle anomalous data points",
      "To increase model complexity",
      "To speed up training",
      "To reduce the number of features"
    ],
    correctAnswer: 0,
    explanation: "Outlier detection helps identify data points that are significantly different from the rest of the data, which can affect model performance.",
    reference: "https://machinelearningmastery.com/model-based-outlier-detection-and-removal-in-python/"
  },
  {
    id: 42,
    question: "What is the purpose of feature importance?",
    options: [
      "To identify the most relevant features for prediction",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "Feature importance helps identify which features contribute most to the model's predictions, aiding in feature selection and model interpretation.",
    reference: "https://machinelearningmastery.com/calculate-feature-importance-with-python/"
  },
  {
    id: 43,
    question: "What is the purpose of model interpretability?",
    options: [
      "To understand how models make predictions",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "Model interpretability helps understand how models make predictions, which is important for trust, debugging, and regulatory compliance.",
    reference: "https://machinelearningmastery.com/interpretability-in-machine-learning/"
  },
  {
    id: 44,
    question: "What is the purpose of SHAP values?",
    options: [
      "To explain individual predictions",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "SHAP values provide a unified measure of feature importance for individual predictions, helping to explain model behavior.",
    reference: "https://machinelearningmastery.com/shap-values-for-feature-selection-in-python/"
  },
  {
    id: 45,
    question: "What is the purpose of LIME?",
    options: [
      "To explain individual predictions",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "LIME (Local Interpretable Model-agnostic Explanations) helps explain individual predictions by approximating the model locally with an interpretable model.",
    reference: "https://machinelearningmastery.com/lime-for-interpretable-machine-learning/"
  },
  {
    id: 46,
    question: "What is the purpose of model deployment?",
    options: [
      "To make models available for use in production",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "Model deployment involves making trained models available for use in production environments, where they can make predictions on new data.",
    reference: "https://machinelearningmastery.com/deploy-machine-learning-models-to-production/"
  },
  {
    id: 47,
    question: "What is the purpose of model monitoring?",
    options: [
      "To track model performance in production",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "Model monitoring helps track model performance in production, identifying issues like concept drift or performance degradation.",
    reference: "https://machinelearningmastery.com/monitor-machine-learning-models-in-production/"
  },
  {
    id: 48,
    question: "What is the purpose of A/B testing?",
    options: [
      "To compare different models or strategies",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "A/B testing helps compare different models or strategies by randomly assigning users to different groups and measuring their performance.",
    reference: "https://machinelearningmastery.com/ab-testing-for-machine-learning-models/"
  },
  {
    id: 49,
    question: "What is the purpose of model versioning?",
    options: [
      "To track different versions of models",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "Model versioning helps track different versions of models, making it easier to roll back changes or compare performance across versions.",
    reference: "https://machinelearningmastery.com/version-control-for-machine-learning-models/"
  },
  {
    id: 50,
    question: "What is the purpose of model serving?",
    options: [
      "To make predictions with deployed models",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "Model serving involves making predictions with deployed models, typically through an API or service that can handle prediction requests.",
    reference: "https://machinelearningmastery.com/serve-machine-learning-models-with-flask/"
  },
  {
    id: 51,
    question: "What is the purpose of model retraining?",
    options: [
      "To update models with new data",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "Model retraining involves updating models with new data to maintain or improve their performance over time.",
    reference: "https://machinelearningmastery.com/retrain-machine-learning-models-with-new-data/"
  },
  {
    id: 52,
    question: "What is the purpose of model evaluation?",
    options: [
      "To assess model performance",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "Model evaluation helps assess how well a model performs on unseen data, using appropriate metrics for the task at hand.",
    reference: "https://machinelearningmastery.com/evaluate-machine-learning-models-with-python/"
  },
  {
    id: 53,
    question: "What is the purpose of hyperparameter tuning?",
    options: [
      "To find optimal model parameters",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "Hyperparameter tuning helps find the optimal values for model parameters that are not learned during training.",
    reference: "https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/"
  },
  {
    id: 54,
    question: "What is the purpose of grid search?",
    options: [
      "To systematically search for optimal hyperparameters",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "Grid search systematically tries different combinations of hyperparameters to find the optimal values.",
    reference: "https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/"
  },
  {
    id: 55,
    question: "What is the purpose of random search?",
    options: [
      "To efficiently search for optimal hyperparameters",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "Random search samples hyperparameter combinations randomly, often finding good solutions more efficiently than grid search.",
    reference: "https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/"
  },
  {
    id: 56,
    question: "What is the purpose of Bayesian optimization?",
    options: [
      "To efficiently search for optimal hyperparameters",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "Bayesian optimization uses probabilistic models to guide the search for optimal hyperparameters, often requiring fewer evaluations than grid or random search.",
    reference: "https://machinelearningmastery.com/what-is-bayesian-optimization/"
  },
  {
    id: 57,
    question: "What is the purpose of ensemble methods?",
    options: [
      "To combine multiple models for better performance",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "Ensemble methods combine multiple models to improve overall performance, often achieving better results than individual models.",
    reference: "https://machinelearningmastery.com/ensemble-methods-for-deep-learning-neural-networks/"
  },
  {
    id: 58,
    question: "What is the purpose of stacking?",
    options: [
      "To combine multiple models using another model",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "Stacking combines multiple models by using another model to learn how to best combine their predictions.",
    reference: "https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/"
  },
  {
    id: 59,
    question: "What is the purpose of boosting?",
    options: [
      "To sequentially improve model performance",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "Boosting sequentially builds models that focus on correcting the errors of previous models, often achieving high performance.",
    reference: "https://machinelearningmastery.com/gentle-introduction-to-the-gradient-boosting-algorithm-for-machine-learning/"
  },
  {
    id: 60,
    question: "What is the purpose of bagging?",
    options: [
      "To reduce variance by averaging multiple models",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "Bagging reduces variance by training multiple models on different subsets of the data and averaging their predictions.",
    reference: "https://machinelearningmastery.com/bagging-and-random-forest-ensemble-algorithms-for-machine-learning/"
  },
  {
    id: 61,
    question: "What is the purpose of random forests?",
    options: [
      "To create an ensemble of decision trees",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "Random forests create an ensemble of decision trees, using both bagging and random feature selection to improve performance.",
    reference: "https://machinelearningmastery.com/bagging-and-random-forest-ensemble-algorithms-for-machine-learning/"
  },
  {
    id: 62,
    question: "What is the purpose of gradient boosting?",
    options: [
      "To sequentially build models that minimize a loss function",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "Gradient boosting sequentially builds models that minimize a loss function, often achieving state-of-the-art performance.",
    reference: "https://machinelearningmastery.com/gentle-introduction-to-the-gradient-boosting-algorithm-for-machine-learning/"
  },
  {
    id: 63,
    question: "What is the purpose of XGBoost?",
    options: [
      "To provide an efficient implementation of gradient boosting",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "XGBoost provides an efficient implementation of gradient boosting with additional features like regularization and parallel processing.",
    reference: "https://machinelearningmastery.com/gentle-introduction-to-xgboost-for-applied-machine-learning/"
  },
  {
    id: 64,
    question: "What is the purpose of LightGBM?",
    options: [
      "To provide a fast and efficient gradient boosting framework",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "LightGBM provides a fast and efficient gradient boosting framework that uses histogram-based algorithms and leaf-wise tree growth.",
    reference: "https://machinelearningmastery.com/gradient-boosting-with-lightgbm/"
  },
  {
    id: 65,
    question: "What is the purpose of CatBoost?",
    options: [
      "To handle categorical features in gradient boosting",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "CatBoost is designed to handle categorical features in gradient boosting, with built-in support for categorical variables and missing values.",
    reference: "https://machinelearningmastery.com/catboost-for-gradient-boosting/"
  },
  {
    id: 66,
    question: "What is the purpose of neural networks?",
    options: [
      "To learn complex patterns in data",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "Neural networks can learn complex patterns in data through multiple layers of interconnected neurons, making them powerful for many machine learning tasks.",
    reference: "https://machinelearningmastery.com/what-are-artificial-neural-networks/"
  },
  {
    id: 67,
    question: "What is the purpose of deep learning?",
    options: [
      "To learn hierarchical representations of data",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "Deep learning uses neural networks with many layers to learn hierarchical representations of data, often achieving state-of-the-art performance.",
    reference: "https://machinelearningmastery.com/what-is-deep-learning/"
  },
  {
    id: 68,
    question: "What is the purpose of transfer learning?",
    options: [
      "To leverage pre-trained models for new tasks",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "Transfer learning allows models to leverage knowledge from pre-trained models, often requiring less data and training time for new tasks.",
    reference: "https://machinelearningmastery.com/transfer-learning-for-deep-learning/"
  },
  {
    id: 69,
    question: "What is the purpose of reinforcement learning?",
    options: [
      "To learn through interaction with an environment",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "Reinforcement learning learns through interaction with an environment, receiving rewards or penalties for actions taken.",
    reference: "https://machinelearningmastery.com/what-is-reinforcement-learning/"
  },
  {
    id: 70,
    question: "What is the purpose of unsupervised learning?",
    options: [
      "To find patterns in unlabeled data",
      "To increase model complexity",
      "To speed up training",
      "To handle missing data"
    ],
    correctAnswer: 0,
    explanation: "Unsupervised learning finds patterns in unlabeled data, helping to discover structure and relationships without predefined outputs.",
    reference: "https://machinelearningmastery.com/what-is-unsupervised-learning/"
  }
]; 