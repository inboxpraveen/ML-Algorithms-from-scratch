import numpy as np

class AdaBoost:
    """
    AdaBoost (Adaptive Boosting) Implementation from Scratch
    
    AdaBoost is an ensemble learning algorithm that combines multiple weak classifiers
    (typically decision stumps) to create a strong classifier. It sequentially trains
    learners, with each new learner focusing on examples that previous learners got wrong.
    
    Key Idea: "Combine weak learners through weighted voting to create a strong learner"
    
    Use Cases:
    - Binary Classification: Face detection, spam filtering
    - Medical Diagnosis: Disease prediction from symptoms
    - Fraud Detection: Identifying fraudulent transactions
    - Customer Analytics: Churn prediction, conversion prediction
    
    Key Concepts:
        Weak Learner: A classifier slightly better than random guessing (e.g., decision stump)
        Sample Weights: Importance of each training example (adaptive)
        Learner Weight (Alpha): How much to trust each weak learner
        Final Prediction: Weighted majority vote of all learners

    AdaBoost Update Rules (the three formulas this class implements):
        1. Weighted error of learner t:
               e_t = sum_i w_i * I[h_t(x_i) != y_i]        (weights sum to 1)
        2. Learner weight (Freund-Schapire convention):
               alpha_t = 0.5 * ln((1 - e_t) / e_t) * learning_rate
        3. Sample re-weighting, then renormalise to sum to 1:
               w_i <- w_i * exp(-alpha_t * y_i * h_t(x_i))
           Because y_i and h_t(x_i) are both in {-1, +1}, the product y_i*h_t(x_i)
           is +1 when correct and -1 when wrong, so this is
               correct -> w_i * e^(-alpha_t)   (weight shrinks)
               wrong   -> w_i * e^(+alpha_t)   (weight grows)
           The ratio between a wrong and a correct sample is e^(2*alpha_t)
           = ((1-e_t)/e_t)^learning_rate. With learning_rate=1 this makes h_t's
           weighted error under the NEW weights exactly 0.5 -- AdaBoost's
           defining invariant, and the reason learner t+1 cannot simply repeat
           learner t.
        4. Final prediction:
               H(x) = sign(sum_t alpha_t * h_t(x)),  ties (sum == 0) resolved to +1

    Label convention: y must contain only -1 and +1. Convert 0/1 labels first
    with y = np.where(y == 0, -1, 1).

    Simplifications vs. canonical AdaBoost (see the .md section of that name):
        - Binary labels only; no SAMME multi-class extension, no SAMME.R,
          no AdaBoost.R2 regression variant.
        - Decision stumps are the only weak learner (no pluggable base
          estimator), and fit() takes no per-sample sample_weight: the
          initial weights are always uniform at 1/N.
        - alpha uses 0.5*ln((1-e)/e), exactly half of sklearn SAMME's
          ln((1-e)/e). Predictions are identical because predict() takes only
          the sign of the weighted sum and predict_proba divides by sum|alpha|.
        - The threshold search is an exhaustive O(N^2 * F) scan per round;
          library implementations pre-sort and reach O(N * F * log N).
        - No per-round sample-weight floor (sklearn clips every weight up to
          machine epsilon at the top of each round; sklearn issue #20320).
          It did not bite in any configuration measured inside the documented
          learning_rate range of 0.1 to 1.0 (three datasets, learning_rate in
          {0.1, 0.5, 1.0}, n_estimators=100: no alpha ever reached the clip).
          Above that range the weights collapse until alpha pins at the value
          the 1e-10 error clip implies. On make_classification(n_samples=150,
          n_features=4, random_state=3) with n_estimators=50 the pinning
          starts at learning_rate=2.0 (18 of 50 alphas pinned, train accuracy
          still 100%) and is ruinous at learning_rate=5.0: train accuracy
          96.67% after the first stump, 4.67% after all 50, 48 of the 50
          alphas pinned at 57.5646. sklearn's AdaBoostClassifier with depth-1
          stumps at the same learning_rate and n_estimators scores 96.67%.
    """
    
    def __init__(self, n_estimators=50, learning_rate=1.0):
        """
        Initialize the AdaBoost classifier
        
        Parameters:
        -----------
        n_estimators : int, default=50
            Number of weak learners to train sequentially
            - More estimators: Better training fit, longer training, risk overfitting
            - Fewer estimators: Faster training, may underfit
            Typical values: 50-200
            
        learning_rate : float, default=1.0
            Shrinks the contribution of each classifier
            - Lower values need more estimators but generalize better
            - learning_rate * n_estimators ~= constant for similar performance
            - Range: 0.1 to 1.0. Not merely advice: this class applies no
              sample-weight floor, so above that range the weights can
              collapse until alpha pins at the error clip (seen from
              learning_rate=2.0 upward) and the ensemble can end up far worse
              than a single stump. See the last bullet of the class
              docstring. Values > 1.0 are not validated or rejected here,
              only unsupported.
            Typical: 0.5-1.0 for small datasets, 0.1-0.3 for large datasets
            Note: the learning rate scales alpha, so the re-weighting ratio
            becomes ((1-e)/e)^learning_rate -- the same semantics as sklearn's
            AdaBoostClassifier(algorithm='SAMME').
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.alphas = []  # Weights for each weak learner
        self.weak_learners = []  # Store trained weak learners
        self.n_features = None  # Set by fit(); also the "is fitted?" flag

    def _as_2d_float(self, X):
        """
        Coerce user input (list, 1-D array, int array) into a 2-D float array

        Keeps fit/predict working for the shapes the docstrings promise:
        a plain Python list of lists, or a 1-D array for a single feature.

        Returns:
        --------
        X : np.ndarray, shape (n_samples, n_features), dtype float
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            # A single feature given as a flat vector -> one column
            X = X.reshape(-1, 1)
        return X

    def _check_is_fitted(self):
        """Raise a clear error instead of a cryptic one when fit() was never called"""
        if self.n_features is None or len(self.weak_learners) == 0:
            raise ValueError(
                "This AdaBoost instance is not fitted yet. "
                "Call fit(X, y) before using predict/predict_proba/score."
            )
        
    def _create_decision_stump(self, X, y, weights):
        """
        Create a decision stump (1-level decision tree)
        
        A decision stump makes predictions based on a single feature threshold:
        - If feature_i <= threshold: predict class_left
        - If feature_i > threshold: predict class_right
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data
        y : np.ndarray, shape (n_samples,)
            Target labels (-1 or +1)
        weights : np.ndarray, shape (n_samples,)
            Sample weights (normalized, sum to 1)
            
        Returns:
        --------
        stump : dict
            Dictionary containing:
            - 'feature': Feature index to split on
            - 'threshold': Threshold value
            - 'left_prediction': Prediction for samples <= threshold
            - 'right_prediction': Prediction for samples > threshold
        error : float
            Weighted classification error of the returned stump.
            Because both polarities are searched, this is always <= half the
            total weight (0.5 for the normalized weights fit() passes):
            a stump can never be worse than random on the weighted data.
        """
        n_samples, n_features = X.shape
        best_error = float('inf')
        best_stump = None

        # Total weight, used by the mirrored-polarity shortcut below.
        # fit() always passes normalized weights, so this is 1.0 up to
        # floating-point rounding -- but summing it here keeps the shortcut
        # correct for any weight vector a caller hands in.
        total_weight = np.sum(weights)

        # Try all features
        for feature_idx in range(n_features):
            # Get unique values for this feature
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)

            # Candidate thresholds are MIDPOINTS between consecutive distinct
            # values, as sklearn's tree splitter does. Splitting on a raw
            # observed value would put the boundary on top of a training point;
            # the midpoint sits in the gap and generalises better. (It also
            # drops one candidate the raw-value list has: threshold = max(v),
            # which sends every sample left and is just a constant classifier.
            # sklearn's depth-1 tree cannot emit that either.)
            if len(unique_values) == 1:
                # Constant feature: no gap to split in. The single candidate
                # still lets the stump act as a constant classifier.
                thresholds = unique_values
            else:
                thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0

                # Guard the midpoint, the same way sklearn's splitter does.
                # For two values one ULP apart the midpoint ROUNDS UP onto the
                # upper value (this happens for half of all adjacent double
                # pairs), and for values near 1e308 the sum overflows to
                # +/-inf. Because the rule below is `col <= threshold`, either
                # case drags the upper value to the LEFT and makes the intended
                # split unreachable: a separable two-value feature would then
                # score 0.5 error and alpha 0. Falling back to the lower value
                # restores the split, and never fires on ordinary data.
                rounded_up = thresholds >= unique_values[1:]
                overflowed = ~np.isfinite(thresholds)
                thresholds = np.where(rounded_up | overflowed,
                                      unique_values[:-1], thresholds)

            # Try all possible thresholds
            for threshold in thresholds:
                # Polarity A: predict -1 for values <= threshold, +1 above it
                predictions = np.ones(n_samples)
                predictions[feature_values <= threshold] = -1
                
                # Calculate weighted error
                misclassified = (predictions != y).astype(float)
                error = np.sum(weights * misclassified)
                
                # Keep track of best split
                if error < best_error:
                    best_error = error
                    best_stump = {
                        'feature': feature_idx,
                        'threshold': threshold,
                        'left_prediction': -1,
                        'right_prediction': 1
                    }
                
                # Polarity B: the mirror image of polarity A. Every sample that
                # polarity A got right, B gets wrong and vice versa, so its
                # weighted error is total_weight - error. (No second pass over
                # the data is needed.)
                # Subtract total_weight, not the literal 1.0. The weights only
                # sum to 1 up to rounding, so `1.0 - error` leaves a residue of
                # up to 2e-16 for a FLAWLESS mirrored stump instead of 0.0, and
                # it is simply wrong if a caller passes unnormalized weights.
                error_flipped = total_weight - error

                if error_flipped < best_error:
                    best_error = error_flipped
                    best_stump = {
                        'feature': feature_idx,
                        'threshold': threshold,
                        'left_prediction': 1,
                        'right_prediction': -1
                    }
        
        return best_stump, best_error
    
    def _stump_predict(self, stump, X):
        """
        Make predictions using a decision stump
        
        Parameters:
        -----------
        stump : dict
            Decision stump parameters
        X : np.ndarray, shape (n_samples, n_features)
            Data to predict
            
        Returns:
        --------
        predictions : np.ndarray, shape (n_samples,)
            Predicted labels (-1 or +1)
        """
        n_samples = X.shape[0]
        feature_values = X[:, stump['feature']]
        
        predictions = np.ones(n_samples) * stump['right_prediction']
        predictions[feature_values <= stump['threshold']] = stump['left_prediction']
        
        return predictions
    
    def fit(self, X, y):
        """
        Train the AdaBoost classifier
        
        Algorithm:
        1. Initialize sample weights equally: w_i = 1/N
        2. For t = 1 to n_estimators:
           a. Train weak learner h_t on weighted data
           b. Calculate weighted error: e_t = sum_i w_i * I[h_t(x_i) != y_i]
           c. Calculate learner weight: alpha_t = 0.5 * ln((1-e_t)/e_t) * learning_rate
           d. Update sample weights: w_i <- w_i * exp(-alpha_t * y_i * h_t(x_i))
              - Correct predictions (y_i * h_t(x_i) = +1): w_i * e^(-alpha_t)
              - Wrong predictions   (y_i * h_t(x_i) = -1): w_i * e^(+alpha_t)
           e. Normalize weights: w_i = w_i / sum_j w_j
           f. If h_t already classifies every training sample correctly,
              boosting has nothing left to correct: store it and stop early
              (this is what sklearn's AdaBoostClassifier does too).
        3. Final model: H(x) = sign(sum_t alpha_t * h_t(x))

        Note on step (d): the exponent uses -alpha * y * h, NOT +alpha * I[wrong].
        The two differ by a square root. Only the form above makes h_t's weighted
        error under the updated weights exactly 0.5, which is what forces round
        t+1 to pick a genuinely different stump.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data. A 1-D array or a plain Python list is accepted and
            treated as a single feature column.
        y : np.ndarray, shape (n_samples,)
            Target labels (must be -1 or +1). For 0/1 labels convert first:
            y = np.where(y == 0, -1, 1)
            
        Returns:
        --------
        self : AdaBoost
            Fitted classifier
        """
        # Validate input
        X = self._as_2d_float(X)
        y = np.asarray(y)
        
        # Ensure labels are -1 and +1
        unique_labels = np.unique(y)
        if not np.all(np.isin(unique_labels, [-1, 1])):
            raise ValueError("Labels must be -1 or +1. Got: {}".format(unique_labels))

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "X and y must have the same number of samples. "
                "Got X: {} and y: {}".format(X.shape[0], y.shape[0])
            )
        
        n_samples, n_features = X.shape
        self.n_features = n_features
        
        # Initialize sample weights uniformly
        weights = np.ones(n_samples) / n_samples
        
        self.alphas = []
        self.weak_learners = []
        
        # Train weak learners sequentially
        for t in range(self.n_estimators):
            # Train decision stump on weighted data
            stump, error = self._create_decision_stump(X, y, weights)
            
            # Prevent error from being 0 or 1 (numerical stability)
            error = np.clip(error, 1e-10, 1 - 1e-10)

            # Calculate learner weight (alpha)
            # Higher alpha = lower error = more trust
            alpha = 0.5 * np.log((1 - error) / error)
            alpha = alpha * self.learning_rate  # Apply learning rate

            # Make predictions with this stump
            predictions = self._stump_predict(stump, X)

            # A perfect stump ends boosting: there are no mistakes left to
            # re-weight, so every sample's weight is multiplied by the same
            # e^(-alpha), normalising back to the weights we came in with --
            # every later round would re-select this identical stump.
            # Test the PREDICTIONS, not the weighted error. The two agree
            # whenever the weights still carry real mass, but at a large
            # learning_rate the easy samples' weights collapse toward 0 and
            # the misclassified mass collapses with them -- long before any
            # weight is literally 0, and it need never get there.
            # On make_moons(n_samples=120, noise=0.25, random_state=1)
            # at learning_rate=3.0 the smallest weight is 4.3e-22 by round 5,
            # where the stump reports a weighted error of 1e-13 while getting
            # 19 of 120 training samples wrong -- so a tolerance like
            # `error <= 1e-10` is no safer. By round 18 the reported error is
            # exactly 0.0 on a stump with 79 of 120 wrong: its true
            # misclassified mass, 9.5e-18, vanishes in the
            # `total_weight - error` subtraction that scores the mirrored
            # polarity. Stopping at round 5 would freeze this run at 84.17%
            # train accuracy and at round 18 at 69.17%, where boosting on to
            # 60 learners reaches 100%.
            # sklearn reaches the same predicate from the other direction: it
            # floors every weight at machine epsilon, so its `error <= 0` test
            # can only fire on a genuinely flawless learner.
            perfect = bool(np.all(predictions == y))

            # Update sample weights: w_i <- w_i * exp(-alpha * y_i * h_t(x_i))
            # y * predictions is +1 where the stump is right, -1 where it is wrong:
            #   Correct: multiply by e^(-alpha) (decrease weight)
            #   Wrong:   multiply by e^(+alpha) (increase weight)
            # This exact form (not exp(+alpha * I[wrong])) is what drives the
            # stump's weighted error to 0.5 under the new weights.
            weights = weights * np.exp(-alpha * y * predictions)
            
            # Normalize weights to sum to 1
            weights = weights / np.sum(weights)
            
            # Store learner and its weight
            self.weak_learners.append(stump)
            self.alphas.append(alpha)
        
            if perfect:
                # Zero training error -- stop early rather than appending
                # n_estimators-1 identical copies of this stump.
                break

        return self
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Combines all weak learners using weighted majority voting:
        H(x) = sign(sum_t alpha_t * h_t(x))
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to predict
            
        Returns:
        --------
        predictions : np.ndarray, shape (n_samples,)
            Predicted labels (-1 or +1). An exact tie (weighted sum == 0) is
            resolved to +1, so the output only ever contains valid class labels.
            np.sign() would return 0 there, which is not a class.
        """
        self._check_is_fitted()
        X = self._as_2d_float(X)
        n_samples = X.shape[0]
        
        # Initialize weighted sum
        weighted_sum = np.zeros(n_samples)
        
        # Add weighted predictions from each learner
        for alpha, stump in zip(self.alphas, self.weak_learners):
            predictions = self._stump_predict(stump, X)
            weighted_sum += alpha * predictions
        
        # Return sign of weighted sum, with ties broken toward +1
        return np.where(weighted_sum >= 0, 1.0, -1.0)
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Converts the weighted sum F(x) = sum_t alpha_t * h_t(x) to a confidence
        score with the SAMME probability transform:

            P(y = +1 | x) = 1 / (1 + exp(-2 * F(x) / sum_t |alpha_t|))
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to predict
            
        Returns:
        --------
        probabilities : np.ndarray, shape (n_samples,)
            Probability of the positive class (+1)
            - Close to 0.12: Strong prediction for class -1
            - Close to 0.50: Uncertain
            - Close to 0.88: Strong prediction for class +1

        Note on the range: F(x) can never exceed sum_t |alpha_t| in magnitude,
        so the exponent is bounded by +/-2 and the output is confined to roughly
        [0.1192, 0.8808] = [1/(1+e^2), 1/(1+e^-2)]. Treat this as a monotone
        confidence score, not a calibrated probability. sklearn's SAMME
        predict_proba has exactly the same bound.

        Note on shape: this returns a 1-D array of P(+1), not the (n_samples, 2)
        matrix that the repo's XGBoost classifier returns. Use
        1 - predict_proba(X) for P(-1).
        """
        self._check_is_fitted()
        X = self._as_2d_float(X)
        n_samples = X.shape[0]
        
        # Calculate weighted sum
        weighted_sum = np.zeros(n_samples)
        for alpha, stump in zip(self.alphas, self.weak_learners):
            predictions = self._stump_predict(stump, X)
            weighted_sum += alpha * predictions
        
        # Convert to probability (sigmoid transformation)
        # Normalize by sum of alphas
        total_alpha = np.sum(np.abs(self.alphas))
        if total_alpha == 0:
            # Every learner scored exactly 0.5 error -> the ensemble carries no
            # information. Report maximum uncertainty instead of dividing by 0.
            return np.full(n_samples, 0.5)
        probabilities = 1 / (1 + np.exp(-2 * weighted_sum / total_alpha))
        
        return probabilities
    
    def score(self, X, y):
        """
        Calculate accuracy on given data
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to evaluate
        y : np.ndarray, shape (n_samples,)
            True labels
            
        Returns:
        --------
        accuracy : float
            Fraction of correct predictions (0 to 1). This is a classifier, so
            score() reports accuracy, not R^2.
        """
        y = np.asarray(y)
        if y.shape[0] == 0:
            raise ValueError("Cannot score on an empty dataset.")
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
    
    def staged_score(self, X, y):
        """
        Calculate accuracy after each weak learner (learning curve)
        
        Useful for:
        - Visualizing training progress
        - Finding optimal number of estimators
        - Detecting overfitting
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to evaluate
        y : np.ndarray, shape (n_samples,)
            True labels
            
        Returns:
        --------
        scores : list of float
            Accuracy after each learner [accuracy_1, accuracy_2, ..., accuracy_T].
            The list has one entry per FITTED learner, which can be fewer than
            n_estimators if fit() stopped early on a perfect stump.
        """
        self._check_is_fitted()
        X = self._as_2d_float(X)
        y = np.asarray(y)
        n_samples = X.shape[0]
        
        scores = []
        weighted_sum = np.zeros(n_samples)
        
        # Incrementally add each learner
        for alpha, stump in zip(self.alphas, self.weak_learners):
            predictions = self._stump_predict(stump, X)
            weighted_sum += alpha * predictions
            
            # Calculate accuracy with learners up to this point.
            # Same tie-break rule as predict(): sum == 0 counts as +1.
            current_predictions = np.where(weighted_sum >= 0, 1.0, -1.0)
            accuracy = np.mean(current_predictions == y)
            scores.append(accuracy)
        
        return scores
    
    def get_feature_importance(self):
        """
        Calculate feature importance
        
        Importance is based on:
        - How often a feature is used for splitting
        - The alpha (weight) of learners that use that feature
        
        Features used by high-alpha learners are more important
        
        Returns:
        --------
        importance : np.ndarray, shape (n_features,)
            Normalized feature importance (sums to 1)
            importance[i] = importance of feature i
        """
        self._check_is_fitted()
        importance = np.zeros(self.n_features)
        
        for alpha, stump in zip(self.alphas, self.weak_learners):
            feature_idx = stump['feature']
            importance[feature_idx] += abs(alpha)
        
        # Normalize
        if np.sum(importance) > 0:
            importance = importance / np.sum(importance)
        
        return importance
    
    def print_learners(self, max_display=10):
        """
        Print information about trained weak learners
        
        Parameters:
        -----------
        max_display : int, default=10
            Maximum number of learners to display

        The 'L->R' column shows the stump's polarity: '-1->+1' means
        "predict -1 at or below the threshold, +1 above it".
        """
        self._check_is_fitted()
        print(f"\n{'='*70}")
        print(f"TRAINED WEAK LEARNERS (showing top {min(max_display, len(self.weak_learners))})")
        print(f"{'='*70}")
        print(f"{'#':>3} {'Feature':>10} {'Threshold':>12} {'Alpha':>12} {'L->R':>10}")
        print(f"{'-'*70}")
        
        for i, (alpha, stump) in enumerate(zip(self.alphas[:max_display], 
                                                self.weak_learners[:max_display])):
            feature = stump['feature']
            threshold = stump['threshold']
            left = stump['left_prediction']
            right = stump['right_prediction']
            direction = f"{left:+d}->{right:+d}"
            
            print(f"{i+1:>3} {feature:>10} {threshold:>12.3f} {alpha:>12.3f} {direction:>10}")
        
        if len(self.weak_learners) > max_display:
            print(f"\n... and {len(self.weak_learners) - max_display} more learners")


"""
USAGE EXAMPLE 1: Simple Binary Classification

import numpy as np

# Create simple 2D dataset
np.random.seed(42)

# Generate linearly separable data
X_class_0 = np.random.randn(50, 2) + np.array([-2, -2])
X_class_1 = np.random.randn(50, 2) + np.array([2, 2])

X = np.vstack([X_class_0, X_class_1])
y = np.array([-1] * 50 + [1] * 50)

# Shuffle data
indices = np.random.permutation(100)
X = X[indices]
y = y[indices]

# Split train/test
X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

# Create and train AdaBoost
model = AdaBoost(n_estimators=50, learning_rate=1.0)
model.fit(X_train, y_train)

# Evaluate
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(f"Training Accuracy: {train_acc:.2%}")
print(f"Test Accuracy: {test_acc:.2%}")

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

print("\nSample Predictions:")
for i in range(5):
    print(f"True: {y_test[i]:+d}, Predicted: {predictions[i]:+.0f}, "
          f"Probability: {probabilities[i]:.3f}")
"""

"""
USAGE EXAMPLE 2: Learning Curves and Model Selection

import numpy as np
import matplotlib.pyplot as plt

# Generate data like Example 1, but with the blob centres at -1/+1 instead
# of -2/+2 so the classes OVERLAP. Example 1's data is separable by a single
# stump, which makes fit() stop after one learner and leaves nothing for a
# learning curve to show.
np.random.seed(42)
X_class_0 = np.random.randn(50, 2) + np.array([-1, -1])
X_class_1 = np.random.randn(50, 2) + np.array([1, 1])
X = np.vstack([X_class_0, X_class_1])
y = np.array([-1] * 50 + [1] * 50)

# Shuffle BEFORE splitting (same as Example 1). The rows were built
# class-sorted, so slicing without a shuffle would put every -1 in train
# and every +1 in test, and the learning curve would be meaningless.
indices = np.random.permutation(100)
X = X[indices]
y = y[indices]

X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

# Train model
model = AdaBoost(n_estimators=100, learning_rate=1.0)
model.fit(X_train, y_train)

# Get learning curves
train_scores = model.staged_score(X_train, y_train)
test_scores = model.staged_score(X_test, y_test)

# Plot learning curves
# Use len(train_scores), not 100: fit() stops early if a stump classifies
# every training sample correctly, so fewer learners may have been trained.
n_fitted = len(train_scores)
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_fitted + 1), train_scores, label='Training', linewidth=2)
plt.plot(range(1, n_fitted + 1), test_scores, label='Testing', linewidth=2)
plt.xlabel('Number of Estimators', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('AdaBoost Learning Curves', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.show()

# Find optimal number of estimators
optimal_n = np.argmax(test_scores) + 1
print(f"Learners actually fitted: {n_fitted} (of 100 requested)")
print(f"Optimal number of estimators: {optimal_n}")
print(f"Best test accuracy: {test_scores[optimal_n-1]:.2%}")
"""

"""
USAGE EXAMPLE 3: Feature Importance

import numpy as np

# Create dataset with 5 features (only first 2 are informative)
np.random.seed(42)
n_samples = 200

# Informative features
X1 = np.random.randn(n_samples, 1)
X2 = np.random.randn(n_samples, 1)

# Non-informative features (noise)
X_noise = np.random.randn(n_samples, 3)

X = np.hstack([X1, X2, X_noise])

# Target depends only on X1 and X2
y = np.where(X1.ravel() + X2.ravel() > 0, 1, -1)

# Train model
model = AdaBoost(n_estimators=50, learning_rate=1.0)
model.fit(X, y)

# Get feature importance
importance = model.get_feature_importance()

print("\nFeature Importance:")
print("="*40)
for i, imp in enumerate(importance):
    print(f"Feature {i}: {imp:.3f} {'#' * int(imp * 50)}")

# Expected: Features 0 and 1 have high importance, rest are low
"""

"""
USAGE EXAMPLE 4: Comparison with Single Decision Stump

import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Generate data a single axis-aligned stump cannot capture but a WEIGHTED SUM
# of stumps can: a ring vs. a disk. The boundary needs several axis-aligned
# cuts stacked together.
#
# Do NOT use XOR here. A stump ensemble is additive, F(x) = A(x0) + B(x1),
# and XOR requires A(+)+B(+) > 0, A(-)+B(-) > 0, A(+)+B(-) < 0, A(-)+B(+) < 0.
# Adding the first two and the last two makes the same quantity both positive
# and negative, so chance accuracy is the best any stump ensemble can do.
np.random.seed(42)

n = 200
X = np.random.randn(n, 2) * 2
y = np.where(X[:, 0] ** 2 + X[:, 1] ** 2 > 4, 1, -1)   # outside the circle vs inside

# Shuffle then split with NO overlap between train and test
indices = np.random.permutation(n)
X, y = X[indices], y[indices]
X_train, X_test = X[:150], X[150:]
y_train, y_test = y[:150], y[150:]

# Train single decision stump
stump = DecisionTreeClassifier(max_depth=1)
stump.fit(X_train, y_train)
stump_acc = stump.score(X_test, y_test)

# Train AdaBoost with multiple stumps
adaboost = AdaBoost(n_estimators=50, learning_rate=1.0)
adaboost.fit(X_train, y_train)
adaboost_acc = adaboost.score(X_test, y_test)

print("Performance Comparison:")
print("="*40)
print(f"Single Decision Stump: {stump_acc:.2%}")
print(f"AdaBoost (50 stumps):  {adaboost_acc:.2%}")
print(f"Improvement:           {(adaboost_acc - stump_acc)*100:.2f} percentage points")

# Show some weak learners
adaboost.print_learners(max_display=5)
"""

"""
USAGE EXAMPLE 5: Effect of Learning Rate

import numpy as np

# Generate data
np.random.seed(42)
X_class_0 = np.random.randn(100, 2) + np.array([-1, -1])
X_class_1 = np.random.randn(100, 2) + np.array([1, 1])
X = np.vstack([X_class_0, X_class_1])
y = np.array([-1] * 100 + [1] * 100)

# Shuffle BEFORE splitting: the rows are class-sorted, so an unshuffled
# split would hand the model one class in train and the other in test.
indices = np.random.permutation(200)
X, y = X[indices], y[indices]

# Non-overlapping split: X[:160] for training, X[160:] for testing.
# (X[:160], X[40:] would put 120 training rows inside the test set and
#  inflate every test number below.)
X_train, X_test = X[:160], X[160:]
y_train, y_test = y[:160], y[160:]

# Try different learning rates, each with the n_estimators budget that
# roughly keeps learning_rate * n_estimators constant
settings = [(0.1, 300), (0.5, 60), (1.0, 30)]

print("Effect of Learning Rate:")
print("="*60)
print(f"{'Learning Rate':>15} {'n_estimators':>15} {'Train Acc':>15} {'Test Acc':>15}")
print("-"*60)

for lr, n_est in settings:
    model = AdaBoost(n_estimators=n_est, learning_rate=lr)
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print(f"{lr:>15.1f} {n_est:>15} {train_acc:>15.2%} {test_acc:>15.2%}")

# Note: learning_rate * n_estimators ~= constant gives comparable models.
#       A low learning rate is not automatically better - it just takes
#       proportionally more estimators to reach the same fit. Run the table
#       yourself: on this easy, well-separated data all three rows land in
#       the same accuracy band.
"""

"""
USAGE EXAMPLE 6: Real-World Application - Spam Detection

import numpy as np

# Simulated email features
# Features: [word_count_free, word_count_click, exclamation_marks, 
#            caps_ratio, link_count, sender_reputation]

np.random.seed(42)

# Spam emails (higher values for suspicious features).
# The spread is 1.5, not 0.5: with a tight spread feature 0 alone separates
# the classes perfectly, fit() stops after one stump, and the importance
# table below would read 1.000 / 0.000 / 0.000 / ... - realistic overlap is
# what makes several features matter.
spam_features = np.random.randn(100, 6) * 1.5 + np.array([3, 2, 4, 0.5, 3, -1])
spam_labels = np.ones(100)

# Ham emails (normal values)
ham_features = np.random.randn(100, 6) * 1.5 + np.array([0.5, 0.2, 0.5, 0.1, 0.5, 1])
ham_labels = np.ones(100) * -1

# Combine data
X = np.vstack([spam_features, ham_features])
y = np.hstack([spam_labels, ham_labels])

# Shuffle
indices = np.random.permutation(200)
X = X[indices]
y = y[indices]

# Split (non-overlapping: 150 train rows, the remaining 50 for test)
X_train, X_test = X[:150], X[150:]
y_train, y_test = y[:150], y[150:]

# Train spam filter
spam_filter = AdaBoost(n_estimators=50, learning_rate=0.8)
spam_filter.fit(X_train, y_train)

# Evaluate
accuracy = spam_filter.score(X_test, y_test)
print(f"\nSpam Filter Accuracy: {accuracy:.2%}")

# Feature importance
feature_names = ['word_free', 'word_click', 'exclamation', 
                'caps_ratio', 'links', 'reputation']
importance = spam_filter.get_feature_importance()

print("\nMost Important Features for Spam Detection:")
for name, imp in sorted(zip(feature_names, importance), 
                       key=lambda x: x[1], reverse=True):
    print(f"  {name:15s}: {imp:.3f}")

# Test on new emails
new_emails = np.array([
    [5, 3, 6, 0.8, 4, -2],  # Likely spam
    [1, 0, 1, 0.05, 1, 2]   # Likely ham
])

predictions = spam_filter.predict(new_emails)
probabilities = spam_filter.predict_proba(new_emails)

print("\nNew Email Classifications:")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    label = "SPAM" if pred == 1 else "HAM"
    print(f"Email {i+1}: {label} (confidence: {prob if pred == 1 else 1-prob:.2%})")
"""

"""
USAGE EXAMPLE 7: Medical Diagnosis

import numpy as np

# Patient features: [age, blood_pressure, cholesterol, bmi, smoking, family_history]
# Target: Heart disease (1) or Healthy (-1)

np.random.seed(42)

# Generate synthetic patient data
# Patients with heart disease (higher risk factors)
diseased = np.random.randn(80, 6) * np.array([10, 15, 20, 3, 0.3, 0.3]) + \
           np.array([65, 150, 240, 30, 0.7, 0.8])

# Healthy patients (lower risk factors)
healthy = np.random.randn(80, 6) * np.array([15, 10, 15, 2, 0.3, 0.3]) + \
          np.array([45, 120, 180, 24, 0.2, 0.3])

X = np.vstack([diseased, healthy])
y = np.array([1] * 80 + [-1] * 80)

# Shuffle
indices = np.random.permutation(160)
X = X[indices]
y = y[indices]

# Split (non-overlapping: 120 train rows, the remaining 40 for test)
X_train, X_test = X[:120], X[120:]
y_train, y_test = y[:120], y[120:]

# Train diagnostic model
model = AdaBoost(n_estimators=30, learning_rate=0.7)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
predictions = model.predict(X_test)

# Calculate precision and recall manually
true_positives = np.sum((predictions == 1) & (y_test == 1))
false_positives = np.sum((predictions == 1) & (y_test == -1))
false_negatives = np.sum((predictions == -1) & (y_test == 1))

precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

print("\nHeart Disease Diagnosis Model:")
print("="*50)
print(f"Accuracy:  {accuracy:.2%}")
print(f"Precision: {precision:.2%} (of predicted diseases, how many correct)")
print(f"Recall:    {recall:.2%} (of actual diseases, how many detected)")

# Feature importance for interpretation
feature_names = ['Age', 'Blood Pressure', 'Cholesterol', 
                'BMI', 'Smoking', 'Family History']
importance = model.get_feature_importance()

print("\nRisk Factor Importance:")
for name, imp in sorted(zip(feature_names, importance), 
                       key=lambda x: x[1], reverse=True):
    print(f"  {name:20s}: {imp:.3f}")

# Diagnose new patients
new_patients = np.array([
    [70, 160, 250, 32, 0.9, 0.9],  # High risk
    [40, 110, 170, 22, 0.0, 0.1]   # Low risk
])

diagnoses = model.predict(new_patients)
probabilities = model.predict_proba(new_patients)

print("\nNew Patient Diagnoses:")
for i, (diag, prob) in enumerate(zip(diagnoses, probabilities)):
    risk = "HIGH RISK" if diag == 1 else "LOW RISK"
    confidence = prob if diag == 1 else 1 - prob
    print(f"Patient {i+1}: {risk} (confidence: {confidence:.2%})")

# Note: This is for educational purposes only!
# Real medical diagnosis requires professional medical evaluation
"""


if __name__ == "__main__":
    # ----------------------------------------------------------------
    # Plug-and-Play Demo: run this file directly with
    #   python _15_adaboost.py
    # numpy only, seeded, ASCII-only output.
    # ----------------------------------------------------------------
    np.random.seed(42)

    # --- Demo 1: binary classification on two Gaussian blobs ---
    print("=" * 55)
    print("DEMO 1 - Binary Classification: two Gaussian blobs")
    print("=" * 55)

    X0 = np.random.randn(100, 2) + np.array([-2, -2])
    X1 = np.random.randn(100, 2) + np.array([2, 2])
    X_cls = np.vstack([X0, X1])
    y_cls = np.array([-1] * 100 + [1] * 100)   # AdaBoost needs -1 / +1 labels

    # Shuffle BEFORE splitting. The rows were stacked class-by-class, so
    # X[:150] without a shuffle would be almost all class -1.
    idx = np.random.permutation(200)
    X_cls, y_cls = X_cls[idx], y_cls[idx]

    # Non-overlapping split: 150 train rows, the other 50 for test.
    X_tr, X_te = X_cls[:150], X_cls[150:]
    y_tr, y_te = y_cls[:150], y_cls[150:]

    clf = AdaBoost(n_estimators=50, learning_rate=1.0)
    clf.fit(X_tr, y_tr)

    print(f"Weak learners fitted : {len(clf.alphas)}")
    print(f"Train Accuracy       : {clf.score(X_tr, y_tr):.2%}")
    print(f"Test  Accuracy       : {clf.score(X_te, y_te):.2%}")

    preds = clf.predict(X_te)
    probs = clf.predict_proba(X_te)
    print("\nSample predictions (true, predicted, P(class=+1)):")
    for i in range(5):
        print(f"  true={int(y_te[i]):+d}  pred={int(preds[i]):+d}  "
              f"P(+1)={probs[i]:.4f}")
    print("Note: P(+1) is bounded to about [0.12, 0.88] by the SAMME transform,")
    print("      so 0.88 already means 'as confident as this model gets'.")

    # --- Demo 2: boosting really does beat a single stump ---
    print("\n" + "=" * 55)
    print("DEMO 2 - One stump vs. 50 boosted stumps (ring vs. disk)")
    print("=" * 55)

    n = 250
    X_ring = np.random.randn(n, 2) * 2
    # Inside the circle -> -1, outside -> +1. One axis-aligned cut cannot
    # describe this, but a weighted SUM of axis-aligned cuts can.
    y_ring = np.where(X_ring[:, 0] ** 2 + X_ring[:, 1] ** 2 > 4, 1, -1)

    idx2 = np.random.permutation(n)
    X_ring, y_ring = X_ring[idx2], y_ring[idx2]
    Xr_tr, Xr_te = X_ring[:190], X_ring[190:]
    yr_tr, yr_te = y_ring[:190], y_ring[190:]

    one = AdaBoost(n_estimators=1).fit(Xr_tr, yr_tr)
    many = AdaBoost(n_estimators=50).fit(Xr_tr, yr_tr)

    print(f"Single stump       -> train {one.score(Xr_tr, yr_tr):.2%}  "
          f"test {one.score(Xr_te, yr_te):.2%}")
    print(f"AdaBoost 50 stumps -> train {many.score(Xr_tr, yr_tr):.2%}  "
          f"test {many.score(Xr_te, yr_te):.2%}")
    gain = (many.score(Xr_te, yr_te) - one.score(Xr_te, yr_te)) * 100
    print(f"Test gain from boosting: {gain:.2f} percentage points")

    staged = many.staged_score(Xr_te, yr_te)
    print("\nTest accuracy as learners are added (first 8 rounds):")
    for i in range(8):
        print(f"  after {i + 1:2d} learner(s): {staged[i]:.2%}")

    # --- Demo 3: feature importance with planted signal ---
    print("\n" + "=" * 55)
    print("DEMO 3 - Feature Importance (2 real features + 3 noise)")
    print("=" * 55)

    n3 = 250
    X_imp = np.random.randn(n3, 5)          # columns 2, 3, 4 are pure noise
    y_imp = np.where(X_imp[:, 0] + X_imp[:, 1] > 0, 1, -1)

    imp_model = AdaBoost(n_estimators=40, learning_rate=1.0)
    imp_model.fit(X_imp, y_imp)

    print(f"Train Accuracy : {imp_model.score(X_imp, y_imp):.2%}")
    print("\nFeature importance (features 0 and 1 should dominate):")
    for i, imp in enumerate(imp_model.get_feature_importance()):
        bar = "#" * int(round(imp * 50))
        print(f"  feature {i}: {imp:.4f} {bar}")

    imp_model.print_learners(max_display=5)
