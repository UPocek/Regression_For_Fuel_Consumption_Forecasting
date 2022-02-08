import numpy as np
from collections import Counter


class Node:
    __slots__ = ['feature', 'threshold', 'data_left', 'data_right', 'gain', 'value']

    def __init__(self, feature=None, threshold=None, data_left=None, data_right=None, gain=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.data_left = data_left
        self.data_right = data_right
        self.gain = gain
        self.value = value


class DecisionTree:
    __slots__ = ['root', 'max_depth', 'min_samples_split']

    def __init__(self, min_samples_split, max_depth):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def _entropy(self, s):
        counts = np.bincount(np.array(s, dtype=np.int64))
        percentages = counts / len(s)
        entropy = 0
        for pct in percentages:
            if pct > 0:
                entropy += pct * np.log2(pct)
        return -entropy

    def _information_gain(self, parent, left_child, right_child):
        num_left = len(left_child) / len(parent)
        num_right = len(right_child) / len(parent)

        return self._entropy(parent) - (num_left * self._entropy(left_child) + num_right * self._entropy(right_child))

    def _best_split(self, X, y):
        best_split = {}
        best_info_gain = -1
        n_rows, n_cols = X.shape

        for f_idx in range(n_cols):
            X_curr = X[:, f_idx]
            for threshold in np.unique(X_curr):
                df = np.concatenate((X, y.reshape(1, -1).T), axis=1)
                df_left = np.array([row for row in df if row[f_idx] <= threshold])
                df_right = np.array([row for row in df if row[f_idx] > threshold])

                if len(df_left) > 0 and len(df_right) > 0:
                    y = df[:, -1]
                    y_left = df_left[:, -1]
                    y_right = df_right[:, -1]

                    gain = self._information_gain(y, y_left, y_right)
                    if gain > best_info_gain:
                        best_split = {
                            'feature_index': f_idx,
                            'threshold': threshold,
                            'df_left': df_left,
                            'df_right': df_right,
                            'gain': gain
                        }
                        best_info_gain = gain
        return best_split

    def _build(self, X, y, depth=0):
        n_rows, n_cols = X.shape

        if n_rows >= self.min_samples_split and depth <= self.max_depth:
            best = self._best_split(X, y)
            if best['gain'] > 0:
                left = self._build(
                    X=best['df_left'][:, :-1],
                    y=best['df_left'][:, -1],
                    depth=depth + 1
                )
                right = self._build(
                    X=best['df_right'][:, :-1],
                    y=best['df_right'][:, -1],
                    depth=depth + 1
                )
                return Node(
                    feature=best['feature_index'],
                    threshold=best['threshold'],
                    data_left=left,
                    data_right=right,
                    gain=best['gain']
                )
        return Node(
            value=Counter(y).most_common(1)[0][0]
        )

    def fit(self, X, y):
        self.root = self._build(X, y)

    def _predict(self, x, tree):
        if tree.value != None:
            return tree.value
        feature_value = x[tree.feature]

        if feature_value <= tree.threshold:
            return self._predict(x=x, tree=tree.data_left)

        if feature_value > tree.threshold:
            return self._predict(x=x, tree=tree.data_right)

    def predict(self, X):
        return [self._predict(x, self.root) for x in X]


class RandomForest:
    __slots__ = ['num_trees', 'min_samples_split', 'max_depth', 'decision_trees', 'preprocessing']

    def __init__(self, num_trees=10, min_samples_split=20, max_depth=4):
        self.num_trees = num_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.decision_trees = []
        self.preprocessing = {"ohe": [], "scaler": []}

    def _sample(self, X, y):
        n_rows, n_cols = X.shape
        samples = np.random.choice(a=n_rows, size=n_rows, replace=True)
        return X[samples], y[samples]

    def fit(self, X, y):
        # Reset
        if len(self.decision_trees) > 0:
            self.decision_trees = []

        num_built = 0
        while num_built < self.num_trees:
            try:
                clf = DecisionTree(
                    min_samples_split=self.min_samples_split,
                    max_depth=self.max_depth
                )
                _X, _y = self._sample(X, y)
                clf.fit(_X, _y)
                self.decision_trees.append(clf)
                num_built += 1
            except Exception:
                continue

    def predict(self, X, y_test):
        y = []
        for tree in self.decision_trees:
            y.append(tree.predict(X))

        y = np.swapaxes(a=y, axis1=0, axis2=1)

        predictions = np.mean(y, axis=1)
        mse = 0
        for r, p in zip(y_test, predictions):
            mse += (r - p) ** 2
        return predictions, mse / len(y_test)

    def make_prediction(self, make, vehicle_class, engine_size, cylinders, transmission, fuel_type, acceleration,
                        weight, horse_power, displacement):
        if make not in ['BMW', 'BUICK', 'CADILLAC', 'CHEVROLET', 'CHRYSLER', 'DODGE',
                        'FIAT', 'FORD', 'HONDA', 'NISSAN', 'SCION', 'TOYOTA', 'VOLKSWAGEN']:
            Exception("Zelejni proizvodjac ne postoji")
        if vehicle_class not in ['COMPACT', 'FULL-SIZE', 'MID-SIZE', 'MINICOMPACT',
                                 'PICKUP TRUCK - STANDARD', 'SUBCOMPACT', 'TWO-SEATER']:
            Exception("Zeljena klasa auta ne postoji")
        if transmission not in ['A4', 'A5', 'A6', 'A8', 'A9', 'AM6', 'AM7', 'AS6', 'AS8', 'AS9',
                                'AV', 'AV7', 'M5', 'M6']:
            Exception("Zeljeni tip prenosa ne postoji")
        if fuel_type not in ['E', 'X', 'Z']:
            Exception("Zeljeni tip goriva ne postoji")

        make = np.array([make]).reshape(-1, 1)
        vehicle_class = np.array([vehicle_class]).reshape(-1, 1)
        engine_size = np.array([engine_size]).reshape(-1, 1)
        cylinders = np.array([cylinders]).reshape(-1, 1)
        transmission = np.array([transmission]).reshape(-1, 1)
        fuel_type = np.array(fuel_type).reshape(-1, 1)
        acceleration = np.array(acceleration).reshape(-1, 1)
        weight = np.array(weight).reshape(-1, 1)
        horsepower = np.array(horse_power).reshape(-1, 1)
        displacement = np.array(displacement).reshape(-1, 1)

        make = self.preprocessing["ohe"][0].transform(make).toarray()
        vehicle_class = self.preprocessing["ohe"][1].transform(vehicle_class).toarray()
        transmission = self.preprocessing["ohe"][2].transform(transmission).toarray()
        fuel_type = self.preprocessing["ohe"][3].transform(fuel_type).toarray()
        engine_size = self.preprocessing["scaler"][0].transform(engine_size)
        cylinders = self.preprocessing["scaler"][1].transform(cylinders)
        acceleration = self.preprocessing["scaler"][2].transform(acceleration)
        weight = self.preprocessing["scaler"][3].transform(weight)
        horsepower = self.preprocessing["scaler"][4].transform(horsepower)
        displacement = self.preprocessing["scaler"][5].transform(displacement)

        data = np.concatenate(
            [make, vehicle_class, engine_size, cylinders, transmission, fuel_type, acceleration, weight, horsepower,
             displacement], axis=1)

        y = []
        for tree in self.decision_trees:
            y.append(tree.predict(data))

        y = np.swapaxes(a=y, axis1=0, axis2=1)

        prediction = np.mean(y, axis=1)

        return prediction[0]
