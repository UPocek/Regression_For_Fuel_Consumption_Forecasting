import numpy as np


class Node:
    __slots__ = ['feature_index', 'threshold', 'left', 'right', 'var_red', 'value']

    def __init__(self, feature_index=None, threshold=None, left=None, right=None, var_red=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.var_red = var_red
        self.value = value


class DecisionTree:
    __slots__ = ['root', 'min_samples_split', 'max_depth', 'preprocessing']

    def __init__(self, min_samples_split=2, max_depth=2):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.preprocessing = {"ohe": [], "scaler": []}

    def build_tree(self, dataset, curr_depth=0):
        X, Y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            best_split = self.get_best_split(dataset, num_features)
            if best_split["var_red"] > 0:
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth + 1)
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth + 1)
                return Node(best_split["feature_index"], best_split["threshold"],
                            left_subtree, right_subtree, best_split["var_red"])

        leaf_value = self.calculate_leaf_value(Y)
        return Node(value=leaf_value)

    def get_best_split(self, dataset, num_features):
        best_split = {}
        max_var_red = -float("inf")
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    curr_var_red = self.variance_reduction(y, left_y, right_y)
                    if curr_var_red > max_var_red:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["var_red"] = curr_var_red
                        max_var_red = curr_var_red

        return best_split

    def split(self, dataset, feature_index, threshold):
        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right

    def variance_reduction(self, parent, l_child, r_child):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        reduction = np.var(parent) - (weight_l * np.var(l_child) + weight_r * np.var(r_child))
        return reduction

    def calculate_leaf_value(self, Y):
        return np.mean(Y)

    def print_tree(self, tree=None, indent=" "):
        if not tree:
            tree = self.root
        if tree.value is not None:
            print(tree.value)
        else:
            print("X_" + str(tree.feature_index), "<=", tree.threshold, "?", tree.var_red)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)

    def fit(self, X, Y):
        X = np.array(X)
        Y = np.array(Y).reshape(-1, 1)
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)

    def _make_one_prediction(self, x, tree):
        if tree.value is not None:
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self._make_one_prediction(x, tree.left)
        else:
            return self._make_one_prediction(x, tree.right)

    def predict(self, X, y_test):
        mse = 0
        predictions = [self._make_one_prediction(x, self.root) for x in X]
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

        prediction = [self._make_one_prediction(x, self.root) for x in data]

        return prediction[0]
