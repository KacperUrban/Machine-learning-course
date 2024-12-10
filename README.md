# Machine-learning-course
# General
This repo is a part of my study course on machine learning.
# Table of content
- [Data preprocessing](#Data-preprocessing)
- [Data classification I](#Data-classification-I)
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
  
Additionaly I have created kedro pipeline. All second notebook I turned into a pipeline with kedro. Flow of data was divide into two part. One is data processing pipeline and second is data science pipeline. In data processing pipeline I did some basic transformation. In data science pipeline is focused on train diffrent models and asses it.

![image](https://github.com/user-attachments/assets/19f8b453-0b1c-4987-a26d-b9be7dc7f2e5)
![image](https://github.com/user-attachments/assets/743c6f3d-2e1b-43d0-afb0-7487215fe618)

# Technologies
- Pandas
- Numpy
- Scikit-learn
- Matplotlib
- Seaborn
- Missingno
- Mlflow
- Kedro
# Status
The project is on going.
