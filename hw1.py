###### Your ID ######
# ID1: 123456789
# ID2: 987654321
#####################

# imports 
import numpy as np
import pandas as pd

def preprocess(X,y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    ###########################################################################
    # TODO: Implement the normalization function.                             #
    ###########################################################################
    # Handle both 1D and 2D arrays for X
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    # Mean normalization: (x - mean) / std
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    # Avoid division by zero
    X_std = np.where(X_std == 0, 1, X_std)
    X = (X - X_mean) / X_std
    
    # Mean normalization for y
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std != 0:
        y = (y - y_mean) / y_std
    
    # If original X was 1D, return it as 1D
    if X.shape[1] == 1:
        X = X.ravel()
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X, y

def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    ###########################################################################
    # TODO: Implement the bias trick by adding a column of ones to the data.  #
    ###########################################################################
    # Handle 1D arrays
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    # Add column of ones at the beginning
    m = X.shape[0]
    ones_column = np.ones((m, 1))
    X = np.hstack([ones_column, X])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X

def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """

    J = 0  # We use J for the cost.
    ###########################################################################
    # TODO: Implement the MSE cost function.                                  #
    ###########################################################################
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    theta = np.asarray(theta, dtype=float).reshape(-1)

    preds = X @ theta
    residuals = preds - y
    m = X.shape[0]
    J = np.mean(residuals ** 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """

    theta = np.asarray(theta, dtype=float).copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    ###########################################################################
    # TODO: Implement the gradient descent optimization algorithm.            #
    ###########################################################################
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    m = X.shape[0]

    for _ in range(num_iters):
        preds = X @ theta
        residuals = preds - y
        # Gradient of MSE (with cost defined as mean squared error)
        grad = (2.0 / m) * (X.T @ residuals)
        theta -= alpha * grad
        J_history.append(compute_cost(X, y, theta))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history

def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """

    pinv_theta = []
    ###########################################################################
    # TODO: Implement the pseudoinverse algorithm.                            #
    ###########################################################################
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    # SVD-based pseudoinverse: X = U * S * V^T  =>  X^+ = V * S^+ * U^T
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    # Invert singular values with tolerance
    tol = np.finfo(float).eps * max(X.shape) * S.max() if S.size > 0 else 0.0
    S_inv = np.array([1/s if s > tol else 0.0 for s in S], dtype=float)
    X_pinv = (VT.T * S_inv) @ U.T  # V * S^+ * U^T
    pinv_theta = X_pinv @ y
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pinv_theta

def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """

    theta = np.asarray(theta, dtype=float).copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    ###########################################################################
    # TODO: Implement the efficient gradient descent optimization algorithm.  #
    ###########################################################################
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    m = X.shape[0]
    prev_J = None
    tol = 1e-8

    for _ in range(num_iters):
        preds = X @ theta
        residuals = preds - y
        grad = (2.0 / m) * (X.T @ residuals)
        theta -= alpha * grad
        J = compute_cost(X, y, theta)
        J_history.append(J)
        if prev_J is not None and abs(prev_J - J) < tol:
            break
        prev_J = J
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history

def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """

    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {} # {alpha_value: validation_loss}
    ###########################################################################
    # TODO: Implement the function and find the best alpha value.             #
    ###########################################################################
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    X_val = np.asarray(X_val, dtype=float)
    y_val = np.asarray(y_val, dtype=float).reshape(-1)

    n = X_train.shape[1]
    for a in alphas:
        theta0 = np.zeros(n, dtype=float)
        theta, _ = efficient_gradient_descent(X_train, y_train, theta0, a, iterations)
        val_loss = compute_cost(X_val, y_val, theta)
        alpha_dict[a] = float(val_loss)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return alpha_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    #####c######################################################################
    # TODO: Implement the function and find the best alpha value.             #
    ###########################################################################
    X_train = np.asarray(X_train, float)
    X_val   = np.asarray(X_val,   float)
    y_train = np.asarray(y_train, float).reshape(-1)
    y_val   = np.asarray(y_val,   float).reshape(-1)

    n_features = X_train.shape[1]
    selected, remaining = [], list(range(n_features))

    # Pre-allocate “bias + selected features” on the fly, adding only 1 column each step
    Xtr_bias = np.ones((X_train.shape[0], 1))
    Xva_bias = np.ones((X_val.shape[0],   1))

    def eval_subset(cols):
        # Stack bias + chosen columns (views; cheap)
        Xtr = np.hstack([Xtr_bias, X_train[:, cols]]) if cols else Xtr_bias
        Xva = np.hstack([Xva_bias, X_val[:, cols]])   if cols else Xva_bias
        theta = compute_pinv(Xtr, y_train)           # OLS closed-form
        return compute_cost(Xva, y_val, theta)

    while len(selected) < min(5, n_features) and remaining:
        best_feat, best_loss = None, np.inf
        # Try adding each remaining feature once (no inner GD loops)
        for f in remaining:
            loss = eval_subset(selected + [f])
            if loss < best_loss:
                best_loss, best_feat = loss, f
        if best_feat is None: break
        selected.append(best_feat)
        remaining.remove(best_feat)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return selected_features

def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    ###########################################################################
    # TODO: Implement the function to add polynomial features                 #
    ###########################################################################
    # Add squared features for numeric columns
    numeric_cols = df_poly.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_poly[f"{col}^2"] = df_poly[col] ** 2
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return df_poly