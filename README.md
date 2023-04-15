# A review of supervised classification techniques in R categorizing short text news headlines

## ABSTRACT
The modern world generates vast amount of information. One of the forms of information we perceive daily is news through a digital or social media. Such vast amount of news can be difficult to parse through for the category in which the user is interested. Categorization of news articles allows users to focus on the news groups they are interested in and allows publications to provide appropriate news to the right reader group. Categorization of numerous articles of varying size, nature, and language can be difficult if done manually. Digital news, being electronically readable, allows us to use machine learning techniques to classify news. This paper examines models of the Machine learning techniques for news classification based on news headlines. Further, the paper also evaluates performance metrics of such techniques on headlines of articles collected from the HuffPost from January 2017 to May 2018. We observe that most techniques examined in this paper do perform better than a null model but are not very accurate at category prediction just based on the short-text headline. However, Random Forest, among all discussed in the paper, provides the best balance for accuracy vs F1 score. In the end we discuss some of the shortcomings, and future improvements.

## Data exploration

#image article count

We’ll work with data collected from the HuffPost (The Huffington Post, n.d.) collected through web scraping and available on Kaggle  (Misra, 2018) . The entire data contains data articles dated from January 2012 to May 2018 with a total count of over 200,000, however for this paper we only take data from January 2017 to May 2018 into account. 

# image categories count
There are 31 distinct categories that house all the articles of the data set. The distribution among those is however imbalanced. The category with the most articles is Politics with 13,680 articles and the one with the least articles is Fifty with only 2 articles. We remove all categories with article count below 100, which brings the distinct categories to 26 and the one with the least count is Science with 135 articles. 

# image headine count
The headline contains descriptive text that can be processed to classify articles. The word count of articles ranges from 1 to 27 with a mean of 10.8939 and median of 11 and SD of 2.78. The distribution when plotted looks normally distributed and QQ plot also suggests the same. 

## Modelling
We will prepare the dataset and split it in a `train_set` and `test_set` with an 80% split with stratified random sampling based on category. We used the initial_split function to achieve the split.  
Figure 5: Code snippet: train - test split
We then created a recipe to apply pre-processing functions to our set before running it through the models. Model is set by calling a model function with parameters, chaining the relevant mode and engine.  
 
Figure 6: Code snippet: tidymodels recipe
Then Model training pipeline is passed as a workflow()and then fitted to the training data. We then defined functions to predict results on the test data, compare it to actual results and collect the metrics and time taken for each model to train such predict. 

