import numpy as np
import math

### Chi square table values ###
chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """Gini impurity of a dataset with labels in the last column."""
    if len(data) == 0:
        return 0.0
    _, counts = np.unique(data[:, -1], return_counts=True)
    p = counts / counts.sum()
    return 1.0 - np.sum(p ** 2)

def calc_entropy(data):
    """Entropy (base 2) of a dataset with labels in the last column."""
    if len(data) == 0:
        return 0.0
    _, counts = np.unique(data[:, -1], return_counts=True)
    p = counts / counts.sum()
    mask = p > 0
    return -np.sum(p[mask] * np.log2(p[mask]))

class DecisionNode:
    def __init__(self, data, impurity_func, feature=-1, depth=0, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data
        self.feature = feature
        self.pred = self.calc_node_pred()
        self.depth = depth
        self.children = []
        self.children_values = []
        self.terminal = False
        self.chi = chi
        self.max_depth = max_depth
        self.impurity_func = impurity_func
        self.gain_ratio = gain_ratio
        self.feature_importance = 0.0

    def calc_node_pred(self):
        if len(self.data) == 0:
            return None
        labels, counts = np.unique(self.data[:, -1], return_counts=True)
        return labels[np.argmax(counts)]

    def add_child(self, node, val):
        self.children.append(node)
        self.children_values.append(val)

    def calc_feature_importance(self, n_total_sample):
        if self.terminal or len(self.children) == 0:
            self.feature_importance = 0.0
            return
        parent_imp = self.impurity_func(self.data)
        m = len(self.data)
        weighted_child_imp = 0.0
        for child in self.children:
            weighted_child_imp += (len(child.data) / m) * self.impurity_func(child.data)
        self.feature_importance = (parent_imp - weighted_child_imp) * (m / n_total_sample)

    def goodness_of_split(self, feature):
        data = self.data
        if len(data) == 0:
            return 0.0, {}
        values = np.unique(data[:, feature])
        groups = {v: data[data[:, feature] == v] for v in values}
        if len(groups) <= 1:
            return 0.0, groups

        m = len(data)
        parent_imp = self.impurity_func(data)
        weighted_child_imp = 0.0
        split_info = 0.0
        
        for v, subset in groups.items():
            w = len(subset) / m
            weighted_child_imp += w * self.impurity_func(subset)
            if self.gain_ratio and w > 0:
                split_info += -w * math.log2(w)
                
        gain = parent_imp - weighted_child_imp
        
        if self.gain_ratio:
            # Use gain ratio
            if self.impurity_func == calc_entropy:
                # Information gain for entropy
                goodness = gain / split_info if split_info > 0 else 0.0
            else:
                # For gini, still use gain ratio approach
                goodness = gain / split_info if split_info > 0 else 0.0
        else:
            goodness = gain
            
        return goodness, groups

    def split(self):
        labels = self.data[:, -1]
        if len(np.unique(labels)) == 1 or self.depth >= self.max_depth:
            self.terminal = True
            return

        n_features = self.data.shape[1] - 1
        best_feat = -1
        best_goodness = -1.0
        best_groups = None
        
        for f in range(n_features):
            goodness, groups = self.goodness_of_split(f)
            if goodness > best_goodness and len(groups) > 1:
                best_goodness = goodness
                best_feat = f
                best_groups = groups

        if best_feat == -1 or best_goodness <= 0:
            self.terminal = True
            return

        # Chi-square pruning
        if self.chi != 1:
            feat_vals = list(best_groups.keys())
            class_vals = list(np.unique(labels))
            r = len(feat_vals)
            c = len(class_vals)
            df = max(1, (r - 1) * (c - 1))

            observed = np.zeros((r, c), dtype=float)
            total = len(self.data)
            
            for i, v in enumerate(feat_vals):
                subset = best_groups[v]
                y, cnts = np.unique(subset[:, -1], return_counts=True)
                for yv, ct in zip(y, cnts):
                    j = class_vals.index(yv)
                    observed[i, j] = ct

            row_sums = observed.sum(axis=1, keepdims=True)
            col_sums = observed.sum(axis=0, keepdims=True)
            expected = row_sums @ col_sums / total
            
            mask = expected > 0
            chi_stat = np.sum(((observed - expected) ** 2)[mask] / expected[mask])

            # Get chi threshold
            if df in chi_table:
                threshold = chi_table[df].get(self.chi, chi_table[df][0.05])
            else:
                # Use the largest df available
                max_df = max([k for k in chi_table.keys() if k <= df]) if any(k <= df for k in chi_table.keys()) else 11
                threshold = chi_table[max_df].get(self.chi, chi_table[max_df][0.05])
                
            if chi_stat < threshold:
                self.terminal = True
                return

        self.feature = best_feat
        for v, subset in best_groups.items():
            child = DecisionNode(
                data=subset,
                impurity_func=self.impurity_func,
                feature=-1,
                depth=self.depth + 1,
                chi=self.chi,
                max_depth=self.max_depth,
                gain_ratio=self.gain_ratio
            )
            self.add_child(child, v)
        self.terminal = False

    def subtree_depth(self):
        if self.terminal or len(self.children) == 0:
            return self.depth
        return max(child.subtree_depth() for child in self.children)

class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data
        self.impurity_func = impurity_func
        self.chi = chi
        self.max_depth = max_depth
        self.gain_ratio = gain_ratio
        self.root = None

    def build_tree(self):
        self.root = DecisionNode(
            data=self.data,
            impurity_func=self.impurity_func,
            feature=-1,
            depth=0,
            chi=self.chi,
            max_depth=self.max_depth,
            gain_ratio=self.gain_ratio
        )
        
        # Build tree recursively
        stack = [self.root]
        while stack:
            node = stack.pop()
            node.split()
            if not node.terminal:
                stack.extend(node.children)
                
        # Calculate feature importance after tree is built
        self._calculate_feature_importance(self.root, len(self.data))
    
    def _calculate_feature_importance(self, node, n_total):
        if node.terminal:
            return
        node.calc_feature_importance(n_total)
        for child in node.children:
            self._calculate_feature_importance(child, n_total)

    def predict(self, instance):
        node = self.root
        while not node.terminal and len(node.children) > 0:
            f = node.feature
            if f == -1:
                break
            val = instance[f]
            try:
                idx = node.children_values.index(val)
                node = node.children[idx]
            except ValueError:
                # Value not seen in training - return majority class
                break
        return node.pred

    def calc_accuracy(self, dataset):
        if len(dataset) == 0:
            return 0.0
        correct = 0
        for i in range(len(dataset)):
            if self.predict(dataset[i]) == dataset[i, -1]:
                correct += 1
        return 100.0 * correct / len(dataset)

def depth_pruning(X_train, X_validation):
    training, validation = [], []
    for max_depth in [1,2,3,4,5,6,7,8,9,10]:
        tree = DecisionTree(X_train, calc_entropy, chi=1, max_depth=max_depth, gain_ratio=True)
        tree.build_tree()
        training.append(tree.calc_accuracy(X_train))
        validation.append(tree.calc_accuracy(X_validation))
    return training, validation

def chi_pruning(X_train, X_test):
    chi_training_acc, chi_validation_acc, depth = [], [], []
    chi_values = [1, 0.5, 0.25, 0.1, 0.05, 0.0001]
    for chi in chi_values:
        tree = DecisionTree(X_train, calc_entropy, chi=chi, max_depth=1000, gain_ratio=True)
        tree.build_tree()
        chi_training_acc.append(tree.calc_accuracy(X_train))
        chi_validation_acc.append(tree.calc_accuracy(X_test))
        depth.append(tree.root.subtree_depth() if tree.root else 0)
    return chi_training_acc, chi_validation_acc, depth

def count_nodes(node):
    if node is None:
        return 0
    total = 1
    for child in node.children:
        total += count_nodes(child)
    return total