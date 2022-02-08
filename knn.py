import numpy as np


class KNN:
    __slots__ = ['sigma', 'mu', 'n_neighbours', 'euclidian_distance', 'X', 'y', 'preprocessing']

    def __init__(self, num_neighbours=6):
        self.sigma, self.mu = None, None
        self.n_neighbours = num_neighbours
        self.euclidian_distance = []
        self.X, self.y = None, None
        self.preprocessing = {"ohe": [], "scaler": []}

    def fit(self, X_train, X_test):
        mu = np.mean(X_train, 0)
        sigma = np.std(X_train, 0)
        self.X = X_train
        X_train = (X_train - mu) / sigma
        X_test = (X_test - mu) / sigma
        for row in range(len(X_test)):
            self.euclidian_distance.append(np.sqrt(np.sum((X_train - X_test[row]) ** 2, axis=1)))

    def predict(self, y_train, y_test):
        mu_y = np.mean(y_train, 0)
        sigma_y = np.std(y_train, 0, ddof=0)
        self.y = y_train
        y_train = (y_train - mu_y) / sigma_y
        y_pred = np.zeros(y_test.shape)
        for row in range(len(self.euclidian_distance)):
            y_pred[row] = y_train[np.argsort(self.euclidian_distance[row], axis=0)[
                                  :int(self.n_neighbours)]].mean() * sigma_y + mu_y
        mse = 0
        for r, p in zip(y_test, y_pred):
            mse += (r - p) ** 2
        return y_pred, mse / len(y_test)

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

        euclidian_distance = np.sqrt(np.sum((self.X - data) ** 2, axis=1))
        prediction = self.y[np.argsort(euclidian_distance, axis=0)[:self.n_neighbours]].mean()

        return prediction
