# Predicting Loan Status using Azure Machine Learning

This project is part of the Udacity Machine Learning with Microsoft Azure Nanodegree. In this project, we built a machine learning model to predict loan status of individuals. We built the model using HyperDrive Config and AUtoML run, and deployed the model as a service.

## Dataset

### Overview

The dataset was obtained from Kaggle - an open source platform for data science competitions. Find the link [here](https://www.kaggle.com/altruistdelhite04/loan-prediction-problem-dataset). The dataset is a Loan prediction dataset, which was used to predict the applicant loan status.

### Task

The dataset contains features like Gender, Marital status, Dependency status, Education, Employement status, Applicant Income, Co-applicant Income, Loan Amount, Credit History, and Property Area. All to predict whether the applicant should receive a loan in the target column - Loan Status.

To build the machine learning model, the dataset was cleaned, using a written function in `train.py` file. This is a step to ensure good metrics score.
After the data preprocessing step, the cleaned dataset will be used in the `hyperparameter step` and `AutoML run` to build the model and obtain an accuracy score from the best run.

### Access

To bring the dataset into the workspace, it was uploaded to `Azure Blob Storage` to retrieve a csv [link](https://cap.blob.core.windows.net/cap/train.csv). A url object was created from here, and passed into the `TabularDatasetFactory` method to retrieve a DataFrame object.

## Automated ML

The AutoMl setting contains parameters as explained below, and the value I choose for each parameter.

`Featurization = auto` - FeaturizationConfig Indicator for whether featurization step should be done automatically or not, or whether customized featurization should be used.

`n_cross_validations = 4 ` - How many cross validations to perform when user validation data is not specified.

`experiment_timeout_minutes" = 30` - Maximum amount of time in hours that all iterations combined can take before the experiment terminates.

`enable_early_stopping" = True` - Whether to enable early termination if the score is not improving in the short term. The default is False.

`verbosity = logging.INFO` - The verbosity level for writing to the log file.

### Results

With an accuracy score of 80%, the best model is VotingEnsemble classifer. After preprocessing, that is, spliting the data into train & test dataset, and concatenating the training data together. The automl config takes the training data, labelled data, cross validation is set to 5. For the model the stopping criteria is at iteration 50 and experiment_timeout_minutes at 30.

**To improve an get better metrics** - Since the data is highly imbalanced, I would explore a method to work with the imbalanced features, and use a performance metrics like `F1Score`.

_TODO_ Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

#### RunDetails - AutoML Best Run Details

![AutoML RunDetails](https://github.com/bleso-a/nd00333-capstone/blob/master/Screenshot/automl%20best%20run.png)

## Hyperparameter Tuning

_TODO_: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

After loading the dataset, it needs to be processed, and the approach is using a function `cleandata` to perform the preprocessing task. To preprocess the categorical features, encoding was done. Then next step splits the data into train and test sets, for the modelling task. These steps were performed in the entry script `train.py` which is the entry script for the SKLearn estimator fed into the HyperDrive config. The role of the HyperDrive is then to vary the **parameters C and max_iter** so that the best performing model is found.

To train the model, the logistic regression algorithm was used, which is an algorithm, for a classification problem.
In order to achieve this, a parameter space and sampling method, an early termination policy, a primary metric name and goal must be provided as part of the hyperdrive config
**The Logisitc regression hyperparameters, `C & max_iter` were utilized.**

**RandomParameterSampling**
The parameter sampler used is the `RandomParameterSampling`, a class that defines random sampling over a hyperparameter search space. The parameter values are choosen from a set of discrete values or a distribution over a continuous range. So this makes the computation less expensive.

**BanditPolicy**
`BanditPolicy`, an early termination policy which is based on `slack factor/slack amount` and `evaluation_interval`. If the primary metric is not within the specified ``slack factor/slack amount`, the policy terminates any runs and this is done with respect to the best performing training run.

**For future work**, it would be nice to explore more into the data, by carrying out data cleaning process and feature engineering activities. Accuracy is not the only evaluation metric process, it would also be nice to explore some other statistical evaluation metrics.

### Results

The accuracy score for this step is 90%. **For future work**, it would be nice to explore more into the data, by carrying out data cleaning process and feature engineering activities. Accuracy is not the only evaluation metric process, it would also be nice to explore some other statistical evaluation metrics.

#### RunDetails - HyperDrive Best Run Details

![RunDetails](https://github.com/bleso-a/nd00333-capstone/blob/master/Screenshot/Best%20run%20with%20parameter.png)

![Run](https://github.com/bleso-a/nd00333-capstone/blob/master/Screenshot/Run%20progress.png)

![BestRun](https://github.com/bleso-a/nd00333-capstone/blob/master/Screenshot/Best%20Run.png)

## Model Deployment - Can be found in the `hyperparameter_tuning.ipynb` notebook

The best model which is from the hyperparameter search is deployed.
In this section, we will talk about the model deployment. The best run of the hyperdrive was deployed, and that was done using `curated_env` object in the notebook to create environment to deploy the already registered model. Using the `Model.deploy()` method the model was deployed, and made available for interaction with an endpoint. Also the `score.py` file which is the entry script receives data submitted to a deployed web service and passes it to the model. It then takes the response returned by the model and returns that to the client.

The two things you need to accomplish in your entry script are:

- Loading the model (using a function called init())
- Running the model on input data (using a function called run())

To interact with the endpoint, a json object was created to send demo data to the endpoint and get prediction results back. Full details in the screen recording.

##### screenshot showing the model endpoint as active.

**Endpoint** - Demo of the deployed model
![Endpoint](https://github.com/bleso-a/nd00333-capstone/blob/master/Screenshot/Endpoint.png)

**Interaction** - Demo of a sample request sent to the endpoint and its response
![Endpoint](https://github.com/bleso-a/nd00333-capstone/blob/master/Screenshot/Interaction.png)

## Screen Recording - To explain the model deployment process.

Watch the [Video](https://youtu.be/0CN12uJnAMA) here
