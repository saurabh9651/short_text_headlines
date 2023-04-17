# News Article Classifier
This repository contains an R script to build various classification models for predicting news article categories based on their headlines. The dataset used is the News Category Dataset v2, which contains articles from various categories, published by HuffPost.

## Getting Started
These instructions will help you set up the project on your local machine.

### Prerequisites
To run the script, you need to have R and RStudio installed on your local machine. You can download and install them from the following links:

- R: https://www.r-project.org/
- RStudio: https://rstudio.com/products/rstudio/download/

### Installing Required Packages
Before running the script, make sure to install the required R packages. You can install them using the following commands in your RStudio console:

```R
install.packages("tidyverse")
install.packages("lubridate")
install.packages("jsonlite")
install.packages("glue")
install.packages("ggplot2")
install.packages("tm")
install.packages("tokenizers")
install.packages("textstem")
install.packages("RSpectra")
install.packages("randomForest")
install.packages("xgboost")
install.packages("caret")
install.packages("parsnip")
install.packages("workflowsets")
install.packages("recipes")
install.packages("yardstick")
install.packages("rsample")
install.packages("keras")
```

### Running the Script
1. Clone the repository to your local machine.
2. Open the R script file (news_category_classification.R) in RStudio.
3. Download dataset from https://arxiv.org/abs/2209.11429
4. Modify the path to the News_Category_Dataset_v2.json file as needed.
5. Run the script to train and evaluate the classification models.

## Description of the Models
The script trains and evaluates multiple classification models, including:
- Null Model
- Naive Bayes
- K-Nearest Neighbors (KNN)
- Decision Trees
- Random Forests
- Support Vector Machines (SVM) with linear and polynomial kernels
- XGBoost
- Multinomial Logistic Regression using Keras

## Evaluation Metrics
The models are evaluated based on the following metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- Area Under the Receiver Operating Characteristic curve (ROC AUC)
- Processing Time
A confusion matrix heatmap is also generated for each model to visualize the model's performance.

## Results
The performance of each model is summarized in a table and visualized in a bar plot. The best model can be identified based on the highest F1 Score, Accuracy, and ROC AUC, as well as the lowest processing time.

## Contributing
Feel free to submit pull requests for any improvements or bug fixes. Please make sure your code follows the same coding style and conventions as the existing code.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
