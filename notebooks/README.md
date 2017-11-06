# GITW 2017 Machine Learning Talk

<div class="vk_c vk_gy vk_sh card-section _MZc">    <div class="vk_bk vk_ans">12:18 PM</div> <div class="vk_gy vk_sh"> Friday, <span class="_Hq">August 25, 2017</span>  <span class="_Hq"> (PDT) </span>  </div> <span>  Time in Seattle, WA  </span>    </div>

***

## Table of Contents

0. Setup ( [notebook](01.0-Preliminaries.ipynb))

1. What is machine learning, and how does it work? ( [notebook](01_machine_learning_intro.ipynb))
    - What is machine learning?
    - What are the two main categories of machine learning?
    - What are some examples of machine learning?
    - How does machine learning "work"?

2. Setting up Python for machine learning: scikit-learn and IPython Notebook ( [notebook](01.1-Machine-Learning-Intro.ipynb))
    - What are the benefits and drawbacks of scikit-learn?
    - How do I install scikit-learn?
    - How do I use the IPython Notebook?
    - What are some good resources for learning Python?

3. Getting started in scikit-learn with the famous iris dataset ([notebook](04.1-Py-Getting-Started-With-Iris-Dataset.ipynb))
    - What is the famous iris dataset, and how does it relate to machine learning?
    - How do we load the iris dataset into scikit-learn?
    - How do we describe a dataset using machine learning terminology?
    - What are scikit-learn's four key requirements for working with data?

4. Training a machine learning model with scikit-learn ([notebook](05-Py-Model-Training.ipynb))
    - What is the K-nearest neighbors classification model?
    - What are the four steps for model training and prediction in scikit-learn?
    - How can I apply this pattern to other machine learning models?

5. Comparing machine learning models in scikit-learn ([notebook](06-Py-Model-Evaluation.ipynb))
    - How do I choose which model to use for my supervised learning task?
    - How do I choose the best tuning parameters for that model?
    - How do I estimate the likely performance of my model on out-of-sample data?

6. Data science pipeline: [pandas](http://pandas.pydata.org/), [seaborn](https://seaborn.pydata.org/), scikit-learn ([notebook](07-Py-Linear-Regression.ipynb))
    - How do I use the pandas library to read data into Python?
    - How do I use the seaborn library to visualize data?
    - What is linear regression, and how does it work?
    - How do I train and interpret a linear regression model in scikit-learn?
    - What are some evaluation metrics for regression problems?
    - How do I choose which features to include in my model?

7. Cross-validation for parameter tuning, model selection, and feature selection ([notebook](08.0-Cross_Validation.ipynb))
    - What is the drawback of using the train/test split procedure for model evaluation?
    - How does K-fold cross-validation overcome this limitation?
    - How can cross-validation be used for selecting tuning parameters, choosing between models, and selecting features?
    - What are some possible improvements to cross-validation?

8. Efficiently searching for optimal tuning parameters ([notebook](09.0-Py-Grid-Search.ipynb))
    - How can K-fold cross-validation be used to search for an optimal tuning parameter?
    - How can this process be made more efficient?
    - How do you search for multiple tuning parameters at once?
    - What do you do with those tuning parameters before making real predictions?
    - How can the computational expense of this process be reduced?

9. Evaluating a classification model ([notebook](10-Py-Classification-Metrics.ipynb))
    - What is the purpose of model evaluation, and what are some common evaluation procedures?
    - What is the usage of classification accuracy, and what are its limitations?
    - How does a confusion matrix describe the performance of a classifier?
    - What metrics can be computed from a confusion matrix?
    - How can you adjust classifier performance by changing the classification threshold?
    - What is the purpose of an ROC curve?
    - How does Area Under the Curve (AUC) differ from classification accuracy?

***

## Examples

### Titanic
- ex.01.0-Titanic-Passengers-Dataset-Explained.ipynb
- ex.01.1-Py-Random-Forest-Titanic.ipynb
- ex.01.2-R-Feature-Engineering-Titanic.ipynb

### Octave
* Food Truck: Linear Regression ([notebook](ex.02-Octave-Linear-Regression-Foodtruck-Profitability.ipynb))
* School Admissions: Logical Regression ([notebook](ex.02-Octave-Logistic-Regression-School-Admissions.ipynb))

### Python Scikit-Learn
* Fruits: Supervised Learning: Decision Tree ([notebook](12.2-Py-Fruit-Decision-Tree.ipynb))
* Iris Flower: Supervised Learning: Decision Tree ([notebook](12.3-Py-Iris-Decision-Tree.ipynb))

***

## All Notebooks

- 01.0-Preliminaries.ipynb
- 01.1-Machine-Learning-Intro.ipynb
- 01.2-ML-Programming-Languages.ipynb
- 01.3-ML-Landscape.ipynb
- 01.4-ML-Landscape-Validate.ipynb
- 01.5-ML-Statistical-Learning.ipynb
- 02.0-ML-Intro-With-Scikit-Learn.ipynb
- 02.1-Basic-Principles.ipynb
- 03-Machine-Learning-Intro.ipynb
- 04.1-Py-Getting-Started-With-Iris-Dataset.ipynb
- 04.2-Py-Iris-Plot-Features.ipynb
- 05-Py-Model-Training.ipynb
- 06-Py-Model-Evaluation.ipynb
- 07-Py-Linear-Regression.ipynb
- 08.0-Cross_Validation.ipynb
- 09.0-Py-Grid-Search.ipynb
- 10-Py-Classification-Metrics.ipynb
- 11-Py-Logistic-Regression-TODO.ipynb
- 12.1-Py-Decision-Tree-TODO.ipynb
- 12.2-Py-Fruit-Decision-Tree.ipynb
- 12.3-Py-Iris-Decision-Tree.ipynb
- 13.1-Py-Random-Decision-Forests.ipynb
- 14.1-Py-Nerual-Networks-TODO.ipynb
- 15.1-Py-Tensorflow-TODO.ipynb
- adv.01-Py-Advanced-Discussion-Classification.ipynb
- adv.02-Py-Training-Linear-Models.ipynb
- adv.03-Py-Support-Vector-Machines.ipynb
- adv.04-Py-Decision-Trees.ipynb
- adv.05-Py-Ensemble-Learning-and-Random-Forests.ipynb
- ex.01.0-Titanic-Passengers-Dataset-Explained.ipynb
- ex.01.1-Py-Random-Forest-Titanic.ipynb
- ex.01.2-R-Feature-Engineering-Titanic.ipynb
- ex.02-Octave-Linear-Regression-Foodtruck-Profitability.ipynb
- ex.03-Python-Housing-Prices.ipynb
- ex.05-Py-Iris-Features-Petal.ipynb
- ex.05-Py-Iris-Features-Sepal.ipynb
- ex.06-Go-kNNclassifier-Iris.ipynb
- test.01-R-Notebook.ipynb
- test.02-Go-Notebook_intro.ipynb