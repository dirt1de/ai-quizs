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
  }
]; 