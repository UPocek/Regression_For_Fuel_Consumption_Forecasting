import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, PowerTransformer
from sklearn.model_selection import train_test_split

import multiple_linear_regression
import ridge_regression
import lasso_regression
import decision_tree
import random_forest
import knn


# - django -

# django-admin startproject myapi
# django-admin startapp myapp
# python manage.py runserver
# python manage.py makemigrations
# python manage.py migrate

# add 'appname' to installed_apps in settings
# create your classes in models.py
# create url endpoints in urls.py
# create views in views.py

# - heroku -

# heroku login
# git init
# heroku git:remote -a nans-project
# pip install gunicorn (gunicorn myapi.wsgi)
# touch Procfile (web: gunicorn myapi.wsgi)
# ALLOWED_HOSTS = ['*'] in django myapi settings.py
# STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
# heroku local -p 4000
# pip freeze > requirements.txt (+ clean up requirements.txt)
# git add .
# git commit -am "make it better"
# git push heroku master

# - fastapi -

# uvicorn app:app --reload
# touch Procfile(web: gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker)
# -||- heroku

def load_data(df, with_print=False, target="fuel_consumption", ):
    make = np.array(df["Make"]).reshape(-1, 1)
    vehicle_class = np.array(df["Vehicle Class"]).reshape(-1, 1)
    engine_size = np.array(df["Engine Size(L)"]).reshape(-1, 1)
    cylinders = np.array(df["Cylinders"]).reshape(-1, 1)
    transmission = np.array(df["Transmission"]).reshape(-1, 1)
    fuel_type = np.array(df["Fuel Type"]).reshape(-1, 1)
    acceleration = np.array(df["acceleration"]).reshape(-1, 1)
    weight = np.array(df["weight"]).reshape(-1, 1)
    horsepower = np.array(df["horsepower"]).reshape(-1, 1)
    displacement = np.array(df["displacement"]).reshape(-1, 1)
    fuel_consumption = np.array(df["Fuel Consumption Comb (L/100 km)"])
    co2_emission = np.array(df["CO2 Emissions(g/km)"])

    ohe1 = OneHotEncoder(handle_unknown='ignore')
    make = ohe1.fit_transform(make).toarray()
    ohe2 = OneHotEncoder(handle_unknown='ignore')
    vehicle_class = ohe2.fit_transform(vehicle_class).toarray()
    ohe3 = OneHotEncoder(handle_unknown='ignore')
    transmission = ohe3.fit_transform(transmission).toarray()
    ohe4 = OneHotEncoder(handle_unknown='ignore')
    fuel_type = ohe4.fit_transform(fuel_type).toarray()

    if with_print:
        print(ohe1.categories_)  # make - 13
        print(ohe2.categories_)  # vehicle_class - 7
        print(ohe3.categories_)  # transmission - 14
        print(ohe4.categories_)  # fuel_type - 3

    data = {}
    data["data"] = np.concatenate(
        [make, vehicle_class, engine_size, cylinders, transmission, fuel_type, acceleration, weight, horsepower,
         displacement], axis=1)
    if target == "fuel_consumption":
        data["target"] = fuel_consumption
    elif target == "co2_emission":
        data["target"] = co2_emission
    return data, (ohe1, ohe2, ohe3, ohe4)


def preprocess_data(X_train, X_test, y_train, y_test, scaler="standard"):
    # later scaling
    if scaler == "minmax":
        # normalizes data by dividing all elements with max (range [0,1])
        scaler1 = MinMaxScaler()
        X_train[:, 20] = scaler1.fit_transform(X_train[:, 20].reshape(-1, 1)).squeeze()
        scaler2 = MinMaxScaler()
        X_train[:, 21] = scaler2.fit_transform(X_train[:, 21].reshape(-1, 1)).squeeze()
        scaler3 = MinMaxScaler()
        X_train[:, 39] = scaler3.fit_transform(X_train[:, 39].reshape(-1, 1)).squeeze()
        scaler4 = MinMaxScaler()
        X_train[:, 40] = scaler4.fit_transform(X_train[:, 40].reshape(-1, 1)).squeeze()
        scaler5 = MinMaxScaler()
        X_train[:, 41] = scaler5.fit_transform(X_train[:, 41].reshape(-1, 1)).squeeze()
        scaler6 = MinMaxScaler()
        X_train[:, 42] = scaler6.fit_transform(X_train[:, 42].reshape(-1, 1)).squeeze()
    elif scaler == "maxabs":
        # normalizes data to range [0,1] (regardless of the sign)
        scaler1 = MaxAbsScaler()
        X_train[:, 20] = scaler1.fit_transform(X_train[:, 20].reshape(-1, 1)).squeeze()
        scaler2 = MaxAbsScaler()
        X_train[:, 21] = scaler2.fit_transform(X_train[:, 21].reshape(-1, 1)).squeeze()
        scaler3 = MaxAbsScaler()
        X_train[:, 39] = scaler3.fit_transform(X_train[:, 39].reshape(-1, 1)).squeeze()
        scaler4 = MaxAbsScaler()
        X_train[:, 40] = scaler4.fit_transform(X_train[:, 40].reshape(-1, 1)).squeeze()
        scaler5 = MaxAbsScaler()
        X_train[:, 41] = scaler5.fit_transform(X_train[:, 41].reshape(-1, 1)).squeeze()
        scaler6 = MaxAbsScaler()
        X_train[:, 42] = scaler6.fit_transform(X_train[:, 42].reshape(-1, 1)).squeeze()
    elif scaler == "robust":
        # smartest version of standardization, works better with outliers then any other
        scaler1 = RobustScaler()
        X_train[:, 20] = scaler1.fit_transform(X_train[:, 20].reshape(-1, 1)).squeeze()
        scaler2 = RobustScaler()
        X_train[:, 21] = scaler2.fit_transform(X_train[:, 21].reshape(-1, 1)).squeeze()
        scaler3 = RobustScaler()
        X_train[:, 39] = scaler3.fit_transform(X_train[:, 39].reshape(-1, 1)).squeeze()
        scaler4 = RobustScaler()
        X_train[:, 40] = scaler4.fit_transform(X_train[:, 40].reshape(-1, 1)).squeeze()
        scaler5 = RobustScaler()
        X_train[:, 41] = scaler5.fit_transform(X_train[:, 41].reshape(-1, 1)).squeeze()
        scaler6 = RobustScaler()
        X_train[:, 42] = scaler6.fit_transform(X_train[:, 42].reshape(-1, 1)).squeeze()
    elif scaler == "powertransformer":
        # applies a power transformation and makes data less subject to outliers and makes data more Gaussian-like
        scaler1 = PowerTransformer(method="yeo-johnson")
        X_train[:, 20] = scaler1.fit_transform(X_train[:, 20].reshape(-1, 1)).squeeze()
        scaler2 = PowerTransformer(method="yeo-johnson")
        X_train[:, 21] = scaler2.fit_transform(X_train[:, 21].reshape(-1, 1)).squeeze()
        scaler3 = PowerTransformer(method="yeo-johnson")
        X_train[:, 39] = scaler3.fit_transform(X_train[:, 39].reshape(-1, 1)).squeeze()
        scaler4 = PowerTransformer(method="yeo-johnson")
        X_train[:, 40] = scaler4.fit_transform(X_train[:, 40].reshape(-1, 1)).squeeze()
        scaler5 = PowerTransformer(method="yeo-johnson")
        X_train[:, 41] = scaler5.fit_transform(X_train[:, 41].reshape(-1, 1)).squeeze()
        scaler6 = PowerTransformer(method="yeo-johnson")
        X_train[:, 42] = scaler6.fit_transform(X_train[:, 42].reshape(-1, 1)).squeeze()
    else:
        scaler1 = StandardScaler()
        X_train[:, 20] = scaler1.fit_transform(X_train[:, 20].reshape(-1, 1)).squeeze()
        scaler2 = StandardScaler()
        X_train[:, 21] = scaler2.fit_transform(X_train[:, 21].reshape(-1, 1)).squeeze()
        scaler3 = StandardScaler()
        X_train[:, 39] = scaler3.fit_transform(X_train[:, 39].reshape(-1, 1)).squeeze()
        scaler4 = StandardScaler()
        X_train[:, 40] = scaler4.fit_transform(X_train[:, 40].reshape(-1, 1)).squeeze()
        scaler5 = StandardScaler()
        X_train[:, 41] = scaler5.fit_transform(X_train[:, 41].reshape(-1, 1)).squeeze()
        scaler6 = StandardScaler()
        X_train[:, 42] = scaler6.fit_transform(X_train[:, 42].reshape(-1, 1)).squeeze()

    X_test[:, 20] = scaler1.transform(X_test[:, 20].reshape(-1, 1)).squeeze()
    X_test[:, 21] = scaler2.transform(X_test[:, 21].reshape(-1, 1)).squeeze()
    X_test[:, 39] = scaler3.transform(X_test[:, 39].reshape(-1, 1)).squeeze()
    X_test[:, 40] = scaler4.transform(X_test[:, 40].reshape(-1, 1)).squeeze()
    X_test[:, 41] = scaler5.transform(X_test[:, 41].reshape(-1, 1)).squeeze()
    X_test[:, 42] = scaler6.transform(X_test[:, 42].reshape(-1, 1)).squeeze()

    return X_train, X_test, y_train, y_test, (scaler1, scaler2, scaler3, scaler4, scaler5, scaler6)


def review_all_models(X_train, X_test, y_train, y_test, plot=False):
    model = multiple_linear_regression.LinearRegression(learning_rate=0.1)  # Test MSE:0.23765791280950466
    model.fit(X_train, y_train)
    predictions, mse = model.predict(X_test, y_test)
    print(f"LinearRegression Test MSE:{mse}")
    if plot:
        plt.plot(np.arange(1, model.iterations + 1, 1), model.loss, label="LinearRegression", color="orange")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        print(f"Final loss:{model.loss[-1]}")
    model = ridge_regression.RidgeRegression(learning_rate=0.1)  # Test MSE:0.24605231657082513
    model.fit(X_train, y_train)
    predictions, mse = model.predict(X_test, y_test)
    print(f"RidgeRegression Test MSE:{mse}")
    if plot:
        plt.plot(np.arange(1, model.iterations + 1, 1), model.loss, label="RidgeRegression", color="blue")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        print(f"Final loss:{model.loss[-1]}")
    model = lasso_regression.LassoRegression(learning_rate=0.1)  # Test MSE:0.2431037561065247
    model.fit(X_train, y_train)
    predictions, mse = model.predict(X_test, y_test)
    print(f"LassoRegression Test MSE:{mse}")
    if plot:
        plt.plot(np.arange(1, model.iterations + 1, 1), model.loss, label="LassoRegression", color="green")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        print(f"Final loss:{model.loss[-1]}")
    model = decision_tree.DecisionTree(min_samples_split=15, max_depth=5)  # Test MSE:0.2842755370631361
    model.fit(X_train, y_train)
    predictions, mse = model.predict(X_test, y_test)
    print(f"DecisionTree Test MSE:{mse}")
    model = random_forest.RandomForest(num_trees=10, min_samples_split=15, max_depth=5)  # Test MSE:0.4635536585365859
    model.fit(X_train, y_train)
    predictions, mse = model.predict(X_test, y_test)
    print(f"RandomForest Test MSE:{mse}")
    model = knn.KNN(num_neighbours=3)  # Test MSE:0.5281571815718158
    model.fit(X_train, X_test)
    predictions, mse = model.predict(y_train, y_test)
    print(f"KNN Test MSE:{mse}")


if __name__ == '__main__':
    df = pd.read_csv('final_ds.csv')
    data, ohe = load_data(df)
    X = data["data"]
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    X_train, X_test, y_train, y_test, scalers = preprocess_data(X_train, X_test, y_train, y_test)

    review_all_models(X_train, X_test, y_train, y_test, plot=False)

    # model = multiple_linear_regression.LinearRegression()
    # model.fit(X_train, y_train)
    # model.predict(X_test, y_test)
    # model.preprocessing = {"ohe": ohe, "scaler": scalers}
    # print(model.make_prediction("BMW", "COMPACT", 2.0, 4, "A8", "Z", 12.8, 2600, 110, 121.0))
    # pickle.dump(model, open("models/linear_regression.pickle", "wb"))
