# Machine-learning-course
# General
This repo is a part of my study course on machine learning.
# Table of content
- [Data preprocessing](#Data-preprocessing)
- [Data classification I](#Data-classification-I)
- [Data classification II](#Data-classification-II)
- [Data classifiction and regression](#Data-classifiction-and-regression)
- [Data classification (Computer vision)](#Data-classification-(Computer-vision))
- [Technologies](#Technologies)
- [Status](#Status)
# Data preprocessing
This task was to get some knowledge about Pokemons dataset. So I have done:
- basic statistics (describe, info etc.)
- basic visualization
- use groupby statements to uncover hidden relationships
- correlation plots
- interpret and handle missing data
- use t-sne to interpret data

Thanks to that I have discovered some inconsistent values (some metrics had a negative values). But overall data was in a good condition. All code you can find in Projekt_nr_1.ipynb.
# Data classification I
That part of the project was mainly focused on to used numeric data to classify name, type 1 and type 2 of the pokemons. So I have done:
- analyze basic statistics
- analyze missing values and remove it
- remove inconsistent data
- encode categorical features
- normalize numerical features
- train few models: Logisitic Regression, SVM, Decision Tree, Random Forest, KNN
- visualize confusion matrix
- calculate few metrics: accuracy, precision, recall, f1-score
- visualize comparison between metrics of diffrent models
  
Additionaly I have created kedro pipeline. All second notebook I turned into a pipeline with kedro. Flow of data was divide into two part. One is data processing pipeline and second is data science pipeline. In data processing pipeline I did some basic transformation. In data science pipeline is focused on train diffrent models and asses it. You can find code in Projekt_nr_2.ipynb.

![image](https://github.com/user-attachments/assets/19f8b453-0b1c-4987-a26d-b9be7dc7f2e5)
![image](https://github.com/user-attachments/assets/743c6f3d-2e1b-43d0-afb0-7487215fe618)

# Data classification II
This phase was build on previous preprocesssing and classification parts. Preprocessing wasn't developt, but I focused on classifiaction. I have done:
- cross validation on different k = 5, 10, 20 (KNN, Logistic regression, Random forest, Decision tree, SVM, Gradient Boosting, Adaboost, Xgboost)
- make data augmention with SMOTE (I know this method mainly is used for balancing classes, but I used it in different manner) to 30, 40 and 50 examples for every class
- check distribution of classes before and after augmention
- feature importance

Findings:
- Random forest achieve the best accuracy (97,8% on k=5)
- Decision tree achieve comparable result to Random forest, but it is much faster method
- best K was 5 and 10
- data augmention increase accuracy
- the most important feature for decision tree was Total, but random forest divide attention equally to every feature

You can find code in Projekt_nr_3.ipynb.

# Data classifiction and regression
In this stage I mainly focused on regression and classification task, but I extened it by dimensionalty reduction methods and I created neural network in PyTorch. I have done:
- use PCA and t-SNE to reduce dimensions,
- in regression problem I used this models: Linear regression, Ridge regression, Lasso, Xgboost, Neural Network,
- in classification problem I used this models: Decision tree, Random forest, Neural Network
- comprehensive summary of classificaton and regression task
- visualize t-SNE transformation

You can find code in Projekt_nr_4.ipynb.

# Data classification (Computer vision)
In this part I used different dataset. It is also about Pokemons, but this time classes are describe as images. I have done:
- preprocess data (create metadata dataframe, remove wrong files, analyse duplicates)
- data exploration and visualization
- create own convolution network in PyTorch
- prepare data for trainig (normalize, resize, crop images etc.)
- introduce data augmenation like ColorJitter, RandomHorizontalFlip etc.
- use pretrained models (ResNet18, EffecientNet)
- comperehensive summary of all results

You can find code in Projekt_nr_5.ipynb.
# Technologies
- Pandas
- Numpy
- Scikit-learn
- Matplotlib
- Seaborn
- Missingno
- Mlflow
- Kedro
- Optuna
- PyTorch

# Status
The project has been completed.
