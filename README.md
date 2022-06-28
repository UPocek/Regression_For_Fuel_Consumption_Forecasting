<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/UPocek/Regression_For_Fuel_Consumption_Forecasting/blob/main/results/front_page.png">
    <img alt="Flutter" src="https://github.com/UPocek/Regression_For_Fuel_Consumption_Forecasting/blob/main/results/front_page.png">
  </picture>
</h1>

# Regression for fuel consumption and C02 emission forecasting

## Introduction and data

To create this project, I decided to use the following 2 datasets:
1. [Auto-mpg dataset](https://www.kaggle.com/datasets/uciml/autompg-dataset)
2. [CO2 Emission by Vehicles](https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles)

which in total contain 21 features, of which 4 are folded. I connected them based on the name of the car (manufacturer + model), which gave me the final dataset with 202 instances, ie. rows and 15 unique columns, of which 2 are predictive (CO2 emissions and fuel consumption during combined driving). In an analysis that will be described in detail later, I selected 10 features that I used to train my models, and as the only non-existent values (for 5 instances) were among the horsepower data, I supplemented them with the mean value for that column.

<img width="1080" alt="image" src="https://user-images.githubusercontent.com/46105849/176285003-f2a06cb4-132c-4d83-ad38-beb4a0de4389.png">
<img width="1080" alt="image" src="https://user-images.githubusercontent.com/46105849/176285065-ca093b53-1864-4e73-a765-54a9cfae750a.png">

## Final data set

The data that I decided to use to train my models were obtained by correlation matrix analysis, empirically as well as using a lasso regression model in order to select features and they are: ['Name', 'Vehicle Class', 'Engine Size (L)', 'Cylinders', 'Transmission', 'Fuel Type', 'acceleration', 'weight', 'horsepower', 'displacement] and I taught them to do prediction for [' Fuel Consumption Comb (L / 100 km) ',' CO2 Emissions (g / km) '].
After dividing the final data set into training and testing (80% -20%), I converted the class data into one-hot-encoding vectors before passing it to the model, while I applied standard scaling to the numerical features to mean zero and standard. division 1 has been shown to give the best results when converting the model and obtaining the least predictive error.

<img width="1080" alt="image" src="https://user-images.githubusercontent.com/46105849/176285294-d2ccde3e-cb85-49db-a9e4-1342a0c5be62.png">
<img width="1080" alt="image" src="https://user-images.githubusercontent.com/46105849/176285347-c90eb143-015e-4ada-9a32-58fd4a5caefc.png">
<img width="1080" alt="image" src="https://user-images.githubusercontent.com/46105849/176285373-bab4afad-db46-4e22-8402-220cf489b937.png">

## Models

### 1. Linear Regression

Multiple linear regression (MLR) is a basic regression technique that uses several different training variables to predict a particular outcome and over data that it has not potentially seen before.
Multiple regression is an extension of linear (SLR) regression that uses only one independent variable to determine prediction.
Formula and calculation of multiple linear regression: yi = β0 + β1 xi1 + β2 xi2 + ... + βp xip + ϵ
Since in practice we can have dozens and even hundreds of independent variables based on which we train the model to do prediction, we do not calculate optimal parameters (weights and bias) analytically but through an iterative gradient descent algorithm.

<img width="500" alt="image" src="https://user-images.githubusercontent.com/46105849/176285737-f7fdb3c9-7109-408b-9d48-79ba9c21772a.png">


### 2. Ridge Regression

Ridge regression is a special extension of ordinary linear regression that introduces L2 regularization. This method of estimating the coefficients of the multiple regression model works very well even in scenarios where the independent variables are highly correlated.
Lambda is a punishment, therefore, by changing the value of alpha, we control the punishment, that is. the strength with which we try to minimize the coefficients. The higher the alpha values, the higher the penalty and thus the smaller the size of the coefficients.

### 3. Lasso Regression

Lasso regression is an extension of linear regression that uses L1 regularization (i.e., shrinkage). This regression algorithm is suitable for data showing high levels of multicollinearity, and can also be used in the parameter selection process.
L1 regularization means that we add a penalty equal to the absolute value of the size of the coefficients. This type of regularization can result in some coefficients becoming zero and being eliminated from the model, which tells us that these features are not important to us.

### 4. Decision Tree

The decision tree is a nonlinear regression algorithm that we use as a tool for predicting categorical as well as continuous values.
We form a tree by a recurrent procedure by dividing the given data, if there is enough of it (more than a threshold), into 2 parts depending on the value of a certain threshold, which we choose so that our split is the best possible, ie. to keep the error to a minimum. If we do not have enough electricity. for split, their mean value becomes the value of a stable leaf, ie. prediction.

<img width="500" alt="image" src="https://user-images.githubusercontent.com/46105849/176286296-cd3675a8-ada1-412a-9852-3b32c8bdaf7c.png">
<img width="500" alt="image" src="https://user-images.githubusercontent.com/46105849/176286334-1b78b838-773d-47ca-9705-6a978c1c4718.png">
<img width="500" alt="image" src="https://user-images.githubusercontent.com/46105849/176286347-98769df8-2c26-4c6a-9b72-d710298aa8cd.png">

### 5. Random Forest

Random Forest regression is a nonlinear regression algorithm that works by constructing several decision trees during training and resulting in a mean or result of voting on the prediction of individual trees. This algorithm corrects the tendency of decision trees to be studied over a training set, especially if we use small values of min_samples_split and max_depth.
It requires more computing power and a larger set of data than all the other algorithms listed so far.

<img width="500" alt="image" src="https://user-images.githubusercontent.com/46105849/176286805-3fd72bf2-ce0c-4ab6-a354-5bf9cb417bdc.png">

### 6. KNN

K Nearest Neighbors is a nonlinear algorithm that predicts new cases based on Euclidean distance, ie. similarities of 2 or more data. The idea is based on the fact that the value of the new element that we want to predict is most likely quite similar to the values of its closest neighbors (ie electricity that have similar parameter values as it). This algorithm can be used for classification problems, but also regression.

<img width="500" alt="image" src="https://user-images.githubusercontent.com/46105849/176286970-f707b0d7-6ad8-41a6-9b1c-1f66fc94f59e.png">

## Results

<img width="1080" alt="image" src="https://user-images.githubusercontent.com/46105849/176287042-48f5b676-2a22-4944-80ca-ab016c570eb4.png">
<img width="500" alt="image" src="https://user-images.githubusercontent.com/46105849/176287102-84646931-ed5a-490c-9f22-4236fa783ea5.png">
<img width="500" alt="image" src="https://user-images.githubusercontent.com/46105849/176287116-0410772f-29b8-4a6e-b297-f89b69376b63.png">
<img width="500" alt="image" src="https://user-images.githubusercontent.com/46105849/176287127-02113083-5701-4cd0-94b6-2f1559f3a4e4.png">

## The best part

The application is hosted on the https://project.mattmarketing.rs/ so go and test the accuracy of the model for your car
