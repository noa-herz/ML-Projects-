import numpy as np

class conditional_independence():

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        self.X_Y = {
            (0, 0): 0.15,
            (0, 1): 0.15,
            (1, 0): 0.2,
            (1, 1): 0.5
        }  # P(X=x, Y=y)

        self.X_C = {
            (0, 0): 0.15,
            (0, 1): 0.15,
            (1, 0): 0.35,
            (1, 1): 0.35
        }  # P(X=x, C=y)

        self.Y_C = {
            (0, 0): 0.15,
            (0, 1): 0.15,
            (1, 0): 0.35,
            (1, 1): 0.35
        }  # P(Y=y, C=c)

        self.X_Y_C = {
            (0, 0, 0): 0.045,
            (0, 0, 1): 0.045,
            (0, 1, 0): 0.105,
            (0, 1, 1): 0.105,
            (1, 0, 0): 0.105,
            (1, 0, 1): 0.105,
            (1, 1, 0): 0.245,
            (1, 1, 1): 0.245,
        }  # P(X=x, Y=y, C=c)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """

        # Taking the information of the objects
        X = self.X
        Y = self.Y
        X_Y = self.X_Y

        # Checking if the condition of dependent variables exists.
        for x, y in X_Y.keys():
            if X[x] * Y[x] == X_Y[(x, y)]:
                return False
        return True

    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndendent
        """

        # Checking if the condition of independent variables exists.
        for x in self.X:
            for y in self.Y:
                for c in self.C:
                    x_y_c = self.X_Y_C[(x, y, c)]
                    x_c = self.X_C[(x, c)]
                    y_c = self.Y_C[(y, c)]
                    if x_y_c is not None and x_c is not None and y_c is not None:
                        if x_y_c != x_c * y_c:
                            return False
        return True


def poisson_log_pmf(k, rate):
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """

    # Creating a variables for the calculation of the log pmf.
    log_p = None
    lambda_pow = rate**k
    e_pow = np.exp(-rate)
    _k = np.math.factorial(k)

    # The calculation.
    log_p = np.log((lambda_pow*e_pow)/_k)
    return log_p

def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """

    # Creating an array that will contain the log likelihood value of rates[i]
    likelihoods = np.zeros(len(rates))

    # Loop that going over the sampels and rates and calculate that log-likelihood value of rates[i].
    for i, rate in enumerate(rates):
        log_likelihood = 0
        for k in samples:
            log_likelihood += poisson_log_pmf(k, rate)
        likelihoods[i] = log_likelihood
    return likelihoods

def possion_iterative_mle(samples, rates):
    """
    samples: set of univariate discrete observations
    rate: a rate to calculate log-likelihood by.

    return: the rate that maximizes the likelihood 
    """
    rate = 0.0

    # Taking the log-likelihood of the samples and rates and taking the max value.
    likelihoods = get_poisson_log_likelihoods(samples, rates)  # might help
    rate = rates[np.argmax(likelihoods)]
    return rate

def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    mean = None

    # Calculate the mean of the samples.
    mean = np.mean(samples)
    return mean


def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """
    p = None

    # Creating a variables for the calculation of the normal pdf.
    square = (x - mean) ** 2
    exp = np.exp(-(square / (2 * (std ** 2))))
    sqr = np.sqrt(2 * (np.pi) * (std) ** 2)

    # Calculate the normal pdf.
    p = exp / sqr
    return p

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class to calculate the parameters for.
        """
        self.mean = None
        self.std = None
        self.data = None
        self.class_data = None

        # The fields of the object.
        self.data = np.copy(dataset)
        class_data = dataset[dataset[:, -1] == class_value][:, :-1]
        self.class_data = class_data
        self.mean = np.mean(class_data, axis=0)
        self.std = np.std(class_data, axis=0)
    
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """

        # Creating a variables for the calculation of the prior probability.
        prior = None
        num_samples = len(self.class_data)
        total_samples = len(self.data)

        # Calculate the prior probability.
        prior = num_samples / total_samples
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """

        # Calculate the likelihood probability.
        likelihood = 1
        for i in range(len(self.data.T) -1):
            likelihood *= normal_pdf(x[i], self.mean[i], self.std[i])
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """

        # Creating a variables for the calculation of the posterior probability.
        posterior = None
        likelihood = self.get_instance_likelihood(x)
        prior = self.get_prior()

        # Calculate the posterior probability.
        posterior = likelihood * prior
        return posterior

class MAPClassifier():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions. 
        One for class 0 and one for class 1, and will predict an instance
        using the class that outputs the highest posterior probability 
        for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods 
                     for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods 
                     for the distribution of class 1.
        """

        # The fields of the object.
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None

        # Creating a variables for the prediction of the posterior probability.
        posterior0 = self.ccd0.get_instance_posterior(x)
        posterior1 = self.ccd1.get_instance_posterior(x)

        # Check if the posterior probability of class 0 is higher
        if posterior0 > posterior1:
            pred = 0
        else:
            pred = 1
        return pred

def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given a test_set using a MAP classifier object.
    
    Input
        - test_set: The test_set for which to compute the accuracy (Numpy array). where the class label is the last column
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / test_set size
    """
    acc = None
    num_correct = 0
    test_set_size = len(test_set)

    # Loop that check if the instance's prediction is correctly classified.
    for instance in test_set:
        true_class = instance[-1]
        predicted_class = map_classifier.predict(instance[:-1])
        if predicted_class == true_class:
            num_correct += 1

    # Calculate the accuracy.
    acc = num_correct / test_set_size
    return acc

def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variable normal desnity function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """

    # Creating a variables for the calculation the normal distribution pdf.
    pdf = None
    n = x.shape[0]
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    diff = (x - mean).reshape((n, 1))
    exponent = -0.5 * diff.T @ inv @ diff

    # Calculate the normal distribution pdf.
    pdf = (1.0 / (np.sqrt((2 * np.pi) ** n * det))) * np.exp(exponent)
    return pdf

class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """

        # The fields of the object.
        self.dataset = dataset
        self.class_value = class_value

        class_data = dataset[dataset[:, -1] == class_value, :-1]

        self.mean = np.mean(class_data, axis=0)
        self.cov = np.cov(class_data.T)
        
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """

        # Creating a variables for the calculation of the prior probability.
        prior = None
        num_samples = (self.dataset[:, -1] == self.class_value).sum()
        total_samples = np.shape(self.mean)[0]

        # Calculate the prior probability.
        prior = num_samples / total_samples
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under the class according to the dataset distribution.
        """

        # Calculate the likelihood probability.
        likelihood = None
        likelihood = multi_normal_pdf(x, self.mean, self.cov)
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """

        # Creating a variables for the calculation of the posterior probability.
        posterior = None
        likelihood = self.get_instance_likelihood(x)
        prior = self.get_prior()

        # Calculate the posterior probability.
        posterior = likelihood * prior
        return posterior

class MaxPrior():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum prior classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest prior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """

        # The fields of the object.
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """

        # Creating a variables for the prediction of the posterior probability.
        pred = None
        prior0 = self.ccd0.get_prior()
        prior1 = self.ccd1.get_prior()

        # Check if the posterior probability of class 0 is higher.
        if prior0 > prior1:
            pred = 0
        else:
            pred = 1
        return pred

class MaxLikelihood():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum Likelihood classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest likelihood probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """

        # The fields of the object.
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """

        # Creating a variables for the prediction of the posterior probability.
        pred = None
        likelihood0 = np.prod(self.ccd0.get_instance_likelihood(x))
        likelihood1 = np.prod(self.ccd1.get_instance_likelihood(x))

        # Check if the posterior probability of class 0 is higher.
        if likelihood0 > likelihood1:
            pred = 0
        else:
            pred = 1

        return pred

EPSILLON = 1e-6 # if a certain value only occurs in the test set, the probability for that value will be EPSILLON.

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with laplace smoothing.
        
        Input
        - dataset: The dataset as a numpy array.
        - class_value: Compute the relevant parameters only for instances from the given class.
        """

        # The fields of the object.
        self.data = np.copy(dataset)
        self.class_value = class_value
        self.class_data = dataset[dataset[:, -1] == class_value]
        self.num_feature = len(self.data.T) - 1
    
    def get_prior(self):
        """
        Returns the prior porbability of the class 
        according to the dataset distribution.
        """

        # Creating a variables for the calculation of the prior probability.
        prior = None
        all_instances = len(self.data)
        class_instances = len(self.class_data)

        # Calculate the prior probability.
        prior = class_instances / all_instances
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under 
        the class according to the dataset distribution.
        """
        likelihood = None
        likelihood = 1.0
        n_i = len(self.class_data)

        # Going over the data and take info that we need for the formula.
        for i, feature in enumerate(self.class_data.T[:-1]):
            V_j = len(set(feature))
            n_i_j = len(feature[x[i] == feature[:]])

            # Calculation of the likelihood
            if n_i + V_j != 0:
                likelihood *= (n_i_j + 1) / (n_i + V_j)
            else:
                likelihood *= (n_i_j + 1) / (n_i + V_j + EPSILLON)
        return likelihood
        
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance 
        under the class according to the dataset distribution.
        * Ignoring p(x)
        """

        # Creating a variables for the calculation of the posterior probability.
        posterior = None
        likelihood = self.get_instance_likelihood(x)
        prior = self.get_prior()

        # Calculate the posterior probability.
        posterior = likelihood * prior
        return posterior


class MAPClassifier_DNB():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """

        # The fields of the object.
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """

        # Creating a variables for the prediction of the posterior probability.
        pred = None
        posterior0 = self.ccd0.get_instance_posterior(x)
        posterior1 = self.ccd1.get_instance_posterior(x)

        # Check if the posterior probability of class 0 is higher.
        if posterior0 > posterior1:
            pred = 0
        else:
            pred = 1

        return pred

    def compute_accuracy(self, test_set):
        """
        Compute the accuracy of a given a testset using a MAP classifier object.

        Input
            - test_set: The test_set for which to compute the accuracy (Numpy array).
        Ouput
            - Accuracy = #Correctly Classified / #test_set size
        """
        acc = None
        num_correct = 0

        # Loop that check if the instance's prediction is correctly classified.
        for instance in test_set:
            true_class = instance[-1]
            predicted_class = self.predict(instance[:-1])
            if predicted_class == true_class:
                num_correct += 1

        # Calculate the accuracy.
        acc = num_correct / test_set.shape[0]
        return acc


