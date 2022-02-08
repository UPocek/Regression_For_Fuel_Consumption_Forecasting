import numpy as np


class LinearRegression:
    __slots__ = ['learning_rate', 'iterations', 'weights', 'bias', 'loss', 'preprocessing']

    def __init__(self, learning_rate=0.1, n_iterations=100):
        self.learning_rate = learning_rate
        self.iterations = n_iterations
        self.weights, self.bias = None, None
        self.loss = []
        self.preprocessing = {"ohe": [], "scaler": []}

    def _mean_squared_error(self, y, y_hat):
        error = 0
        for i in range(len(y)):
            error += (y[i] - y_hat[i]) ** 2
        return error / len(y)

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        omega1 = 0.2
        omega2 = 0.8
        m_w = np.zeros(X.shape[1])
        v_w = np.zeros(X.shape[1])
        m_b = 0
        v_b = 0

        for i in range(self.iterations):
            y_hat = np.dot(X, self.weights) + self.bias
            loss = self._mean_squared_error(y, y_hat)
            self.loss.append(loss)

            partial_w = (1 / X.shape[0]) * (2 * np.dot(X.T, (y_hat - y)))
            partial_d = (1 / X.shape[0]) * (2 * np.sum(y_hat - y))

            m_w = omega1 * m_w + (1 - omega1) * partial_w
            v_w = omega2 * v_w + (1 - omega2) * partial_w ** 2
            m_b = omega1 * m_b + (1 - omega1) * partial_d
            v_b = omega2 * v_b + (1 - omega2) * partial_d ** 2

            m_w_hat = m_w / (1-omega1)
            v_w_hat = v_w / (1-omega2)
            m_b_hat = m_b / (1-omega1)
            v_b_hat = v_b / (1-omega2)

            self.weights -= self.learning_rate / np.sqrt(v_w_hat + 1e-8) * m_w_hat
            self.bias -= self.learning_rate / np.sqrt(v_b_hat + 1e-8) * m_b_hat

    def predict(self, X, y_test):
        mse = 0
        predictions = np.dot(X, self.weights) + self.bias
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

        prediction = np.dot(data, self.weights) + self.bias

        return prediction[0]
