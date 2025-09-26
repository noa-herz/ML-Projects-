import numpy as np
import pandas as pd

def pearson_correlation(x, y):
    """
    Calculate the Pearson correlation coefficient for two given columns of data.
    """
    x = np.array(x)
    y = np.array(y)
    
    # Calculate means
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # Calculate numerator
    numerator = np.sum((x - mean_x) * (y - mean_y))
    
    # Calculate denominator
    denominator = np.sqrt(np.sum((x - mean_x) ** 2) * np.sum((y - mean_y) ** 2))
    
    if denominator == 0:
        return 0.0
    
    r = numerator / denominator
    return r

def feature_selection(X, y, n_features=5):
    """
    Select the best features using pearson correlation.
    """
    correlations = {}
    
    # Convert to DataFrame if needed
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X_array = X.values
    else:
        X_array = X
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Calculate correlation for each feature with target
    for i, feature_name in enumerate(feature_names):
        corr = abs(pearson_correlation(X_array[:, i], y))
        correlations[feature_name] = corr
    
    # Sort by correlation and select top n
    sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    best_features = [feature[0] for feature in sorted_features[:n_features]]
    
    return best_features

class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.
    """
    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state
        self.theta = None
        self.Js = []
        self.thetas = []

    def sigmoid(self, z):
        """Sigmoid function"""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def cost_function(self, X, y, theta):
        """Calculate cost function"""
        m = len(y)
        z = X @ theta
        h = self.sigmoid(z)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-7
        h = np.clip(h, epsilon, 1 - epsilon)
        
        cost = (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        return cost

    def fit(self, X, y):
        """
        Fit training data using gradient descent.
        """
        np.random.seed(self.random_state)
        
        # Add intercept term
        m, n = X.shape
        X = np.column_stack([np.ones(m), X])
        
        # Initialize theta
        self.theta = np.random.randn(n + 1) * 0.01
        
        # Reshape y to column vector
        y = y.reshape(-1, 1)
        
        # Gradient descent
        for iteration in range(self.n_iter):
            # Calculate predictions
            z = X @ self.theta.reshape(-1, 1)
            h = self.sigmoid(z)
            
            # Calculate gradient
            gradient = (1/m) * X.T @ (h - y)
            
            # Update theta
            self.theta = self.theta.reshape(-1, 1) - self.eta * gradient
            self.theta = self.theta.flatten()
            
            # Store theta and cost
            self.thetas.append(self.theta.copy())
            cost = self.cost_function(X, y, self.theta.reshape(-1, 1))
            self.Js.append(cost)
            
            # Check convergence
            if iteration > 0 and abs(self.Js[-1] - self.Js[-2]) < self.eps:
                break

    def predict(self, X):
        """
        Return the predicted class labels for given instances.
        """
        # Add intercept term
        m = X.shape[0]
        X = np.column_stack([np.ones(m), X])
        
        # Calculate predictions
        z = X @ self.theta.reshape(-1, 1)
        h = self.sigmoid(z)
        
        # Convert probabilities to class predictions
        preds = (h >= 0.5).astype(int).flatten()
        return preds

def cross_validation(X, y, folds, algo, random_state):
    """
    Perform k-fold cross validation.
    """
    np.random.seed(random_state)
    
    # Create a copy and shuffle
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    # Create folds
    fold_size = n_samples // folds
    accuracies = []
    
    for i in range(folds):
        # Define validation indices
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < folds - 1 else n_samples
        
        # Split data
        val_indices = list(range(val_start, val_end))
        train_indices = list(range(0, val_start)) + list(range(val_end, n_samples))
        
        X_train_fold = X_shuffled[train_indices]
        y_train_fold = y_shuffled[train_indices]
        X_val_fold = X_shuffled[val_indices]
        y_val_fold = y_shuffled[val_indices]
        
        # Train model
        algo.fit(X_train_fold, y_train_fold)
        
        # Predict and calculate accuracy
        predictions = algo.predict(X_val_fold)
        accuracy = np.mean(predictions == y_val_fold)
        accuracies.append(accuracy)
    
    # Return average accuracy
    cv_accuracy = np.mean(accuracies)
    return cv_accuracy

def norm_pdf(data, mu, sigma):
    """
    Calculate normal density function.
    """
    if sigma <= 0:
        return np.zeros_like(data)
    
    coefficient = 1.0 / (sigma * np.sqrt(2 * np.pi))
    exponent = -0.5 * ((data - mu) / sigma) ** 2
    p = coefficient * np.exp(exponent)
    return p

class EM(object):
    """
    Expectation Maximization for Gaussian Mixture Model.
    """
    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state
        np.random.seed(self.random_state)
        
        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = []

    def init_params(self, data):
        """Initialize distribution parameters"""
        n_samples = data.shape[0]
        
        # Initialize weights uniformly
        self.weights = np.ones(self.k) / self.k
        
        # Initialize means randomly from data range
        data_min = np.min(data)
        data_max = np.max(data)
        self.mus = np.random.uniform(data_min, data_max, self.k)
        
        # Initialize standard deviations
        self.sigmas = np.ones(self.k) * np.std(data)
        
        # Initialize responsibilities
        self.responsibilities = np.zeros((n_samples, self.k))

    def expectation(self, data):
        """E step - calculate responsibilities"""
        n_samples = data.shape[0]
        
        # Calculate likelihood for each component
        for j in range(self.k):
            self.responsibilities[:, j] = self.weights[j] * norm_pdf(data.flatten(), 
                                                                     self.mus[j], 
                                                                     self.sigmas[j])
        
        # Normalize responsibilities
        row_sums = np.sum(self.responsibilities, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1e-10  # Avoid division by zero
        self.responsibilities = self.responsibilities / row_sums

    def maximization(self, data):
        """M step - update distribution parameters"""
        n_samples = data.shape[0]
        data = data.flatten()
        
        for j in range(self.k):
            # Effective number of points assigned to gaussian j
            Nj = np.sum(self.responsibilities[:, j])
            
            if Nj > 1e-10:
                # Update weight
                self.weights[j] = Nj / n_samples
                
                # Update mean
                self.mus[j] = np.sum(self.responsibilities[:, j] * data) / Nj
                
                # Update standard deviation
                variance = np.sum(self.responsibilities[:, j] * (data - self.mus[j])**2) / Nj
                self.sigmas[j] = np.sqrt(variance + 1e-10)  # Add small value for stability
            else:
                # Handle empty cluster
                self.weights[j] = 0.001
                self.mus[j] = np.random.uniform(np.min(data), np.max(data))
                self.sigmas[j] = np.std(data)

    def calculate_cost(self, data):
        """Calculate negative log likelihood"""
        data = data.flatten()
        likelihood = np.zeros(len(data))
        
        for j in range(self.k):
            likelihood += self.weights[j] * norm_pdf(data, self.mus[j], self.sigmas[j])
        
        # Avoid log(0)
        likelihood = np.maximum(likelihood, 1e-10)
        return -np.sum(np.log(likelihood))

    def fit(self, data):
        """Fit GMM using EM algorithm"""
        # Initialize parameters
        self.init_params(data)
        
        # EM iterations
        for iteration in range(self.n_iter):
            # E step
            self.expectation(data)
            
            # M step
            self.maximization(data)
            
            # Calculate cost
            cost = self.calculate_cost(data)
            self.costs.append(cost)
            
            # Check convergence
            if iteration > 0 and abs(self.costs[-1] - self.costs[-2]) < self.eps:
                break

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas

def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate GMM density function.
    """
    pdf = np.zeros_like(data, dtype=float)
    
    for i in range(len(weights)):
        pdf += weights[i] * norm_pdf(data, mus[i], sigmas[i])
    
    return pdf

class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model.
    """
    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = {}
        self.em_models = {}
        self.classes = None

    def fit(self, X, y):
        """Fit training data"""
        self.classes = np.unique(y)
        n_samples = len(y)
        
        # Calculate priors
        for c in self.classes:
            self.prior[c] = np.sum(y == c) / n_samples
        
        # Fit EM for each feature and class
        self.em_models = {c: [] for c in self.classes}
        
        for c in self.classes:
            # Get data for this class
            X_c = X[y == c]
            
            # Fit EM for each feature
            for feature_idx in range(X.shape[1]):
                em = EM(k=self.k, random_state=self.random_state)
                em.fit(X_c[:, feature_idx].reshape(-1, 1))
                self.em_models[c].append(em)

    def predict(self, X):
        """Predict class labels"""
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        
        for i in range(n_samples):
            posteriors = {}
            
            for c in self.classes:
                # Start with prior
                posterior = np.log(self.prior[c])
                
                # Multiply by likelihood for each feature (add in log space)
                for feature_idx in range(X.shape[1]):
                    em = self.em_models[c][feature_idx]
                    weights, mus, sigmas = em.get_dist_params()
                    
                    # Calculate likelihood using GMM
                    likelihood = gmm_pdf(np.array([X[i, feature_idx]]), weights, mus, sigmas)
                    likelihood = np.maximum(likelihood, 1e-10)  # Avoid log(0)
                    posterior += np.log(likelihood[0])
                
                posteriors[c] = posterior
            
            # Choose class with highest posterior
            predictions[i] = max(posteriors, key=posteriors.get)
        
        return predictions.astype(int)

def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    """
    Evaluate Logistic Regression and Naive Bayes models.
    """
    # Train Logistic Regression
    lor = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    lor.fit(x_train, y_train)
    
    # Calculate LOR accuracies
    lor_train_pred = lor.predict(x_train)
    lor_test_pred = lor.predict(x_test)
    lor_train_acc = np.mean(lor_train_pred == y_train)
    lor_test_acc = np.mean(lor_test_pred == y_test)
    
    # Train Naive Bayes with GMM
    nb = NaiveBayesGaussian(k=k)
    nb.fit(x_train, y_train)
    
    # Calculate NB accuracies
    nb_train_pred = nb.predict(x_train)
    nb_test_pred = nb.predict(x_test)
    bayes_train_acc = np.mean(nb_train_pred == y_train)
    bayes_test_acc = np.mean(nb_test_pred == y_test)
    
    return {
        'lor_train_acc': lor_train_acc,
        'lor_test_acc': lor_test_acc,
        'bayes_train_acc': bayes_train_acc,
        'bayes_test_acc': bayes_test_acc
    }

def generate_datasets():
    """
    Generate two datasets where one favors Naive Bayes and the other favors Logistic Regression.
    """
    from scipy.stats import multivariate_normal
    np.random.seed(42)
    
    # Dataset A: Multiple well-separated Gaussians per class (favors Naive Bayes)
    # Class 0: Two distinct clusters
    mean1_c0 = [-3, -3, -3]
    mean2_c0 = [3, 3, 3]
    cov_c0 = np.eye(3) * 0.5
    
    data_c0_1 = multivariate_normal.rvs(mean1_c0, cov_c0, 250)
    data_c0_2 = multivariate_normal.rvs(mean2_c0, cov_c0, 250)
    data_c0 = np.vstack([data_c0_1, data_c0_2])
    
    # Class 1: Two distinct clusters
    mean1_c1 = [-3, 3, -3]
    mean2_c1 = [3, -3, 3]
    cov_c1 = np.eye(3) * 0.5
    
    data_c1_1 = multivariate_normal.rvs(mean1_c1, cov_c1, 250)
    data_c1_2 = multivariate_normal.rvs(mean2_c1, cov_c1, 250)
    data_c1 = np.vstack([data_c1_1, data_c1_2])
    
    dataset_a_features = np.vstack([data_c0, data_c1])
    dataset_a_labels = np.hstack([np.zeros(500), np.ones(500)])
    
    # Dataset B: Linearly separable classes (favors Logistic Regression)
    # Class 0
    mean_b0 = [-1, -1, -1]
    cov_b0 = np.eye(3) * 2
    data_b0 = multivariate_normal.rvs(mean_b0, cov_b0, 500)
    
    # Class 1
    mean_b1 = [1, 1, 1]
    cov_b1 = np.eye(3) * 2
    data_b1 = multivariate_normal.rvs(mean_b1, cov_b1, 500)
    
    dataset_b_features = np.vstack([data_b0, data_b1])
    dataset_b_labels = np.hstack([np.zeros(500), np.ones(500)])
    
    return {
        'dataset_a_features': dataset_a_features,
        'dataset_a_labels': dataset_a_labels,
        'dataset_b_features': dataset_b_features,
        'dataset_b_labels': dataset_b_labels
    }