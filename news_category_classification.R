############## Install libraries
install.packages("jsonlite")
install.packages("lubridate")
install.packages("dplyr")
install.packages("stringr")
install.packages("data.table")
install.packages("zoo")
install.packages("ggplot2")
install.packages("tm")
install.packages("textstem")
install.packages("glue")
install.packages("pillar")
install.packages("rlang")
install.packages("dplyr")
install.packages("rstudioapi")
install.packages("tidyverse")
install.packages("tidymodels")
install.packages("tidytext")
install.packages("textrecipes")
install.packages("rsample")
install.packages("discrim")
install.packages("naivebayes")
install.packages("themis")
install.packages("C50")
install.packages("randomForest")
install.packages("LiblineaR")
#install.packages("glmnet")
install.packages("kknn")
install.packages("nnet")
install.packages("CatEncoders")
install.packages("kernlab")
install.packages("xgboost")
install.packages(c("keras","tensorflow"))
install.packages("writexl")
install.packages("findpython")
install.packages("reticulate")
library(findpython)
library(reticulate)
path_to_python <- find_python_cmd()
virtualenv_create("r-reticulate", python = path_to_python)
##library(tensorflow)
#install_tensorflow(envname = "r-reticulate")
#library(keras)
#install_keras(envname = "r-reticulate")
# NOte: Keras will try to install the tensorflow package in Python.
##  If the host doesn't have python or the package installed. 
##  After the installation is done. It will restart the session
### After restart please run the code below the line

#install.packages("tibble")
#install.packages("textdata")
#install.packages("word2vec")


############## Import libraries
library("tidyverse")
library("tidymodels")
library("tidytext")
library("textrecipes")
library("jsonlite")
library("lubridate")
library("dplyr")
library("stringr")
library("data.table")
library("zoo")
library("ggplot2")
library("tm")
library("textstem")
library("glue")
library("rlang")
library("rstudioapi")
library("rsample")
library("discrim")
library("workflows")
library("naivebayes")
library("themis")
library("C50")
library("randomForest")
library("LiblineaR")
#library("glmnet")
library("keras")
library("tensorflow")
library("nnet")
library("CatEncoders")
library("kernlab")
library("xgboost")
library("writexl")
library("findpython")
library("reticulate")
#library(tibble)
#library(textdata)
#library(word2vec)
############### Set seed for random splits
set.seed(1234)

############## Import data
start_time <- Sys.time()
# Set path where the file is stored
## Please make sure that the data file is in the same folder as the R script file
json_file   = glue(dirname(rstudioapi::getSourceEditorContext()$path),'/News_Category_Dataset_v2.json')
# Import JSON data
myData      = fromJSON(sprintf("[%s]", paste(readLines(json_file),collapse=",")))
# Convert JSON data to dataframe.
news_frame   = as.data.frame(myData)
# Print top 6 rows
#head(news_frame)

############### Data exploration

# Print the data structure
str(news_frame)

# Firstly we'll need to convert date column from "chr" to "Date" format
news_frame$date = ymd(news_frame$date)

# Examine structure of the Date column
str(news_frame$date)

# Get the oldest and most recent date
news_frame %>% 
  # find min and max
  summarise(oldest_date = min(date),
            most_recent = max(date))

### As we see the data is rather large and spans back till 2012. 
###   Lets take data recent than 2017 to shorten our train set.
news_frame = filter(news_frame, date >= "2017-01-01")

# Create a varibale that groups all articles and sums the count by year

periodic_articles = news_frame %>% 
  mutate(Months = format(date, "%Y-%m"),   # Create a variable that gets the Year from date
         ncount = 1) %>%               # Put a count var on each row
  group_by(Months) %>%                  # Then group the counts by Year
  summarise(Articles=sum(ncount))      # Summarise with Articles providing the 
#grouped count

# Plot to see the results
ggplot(periodic_articles, aes(y = Months, x = Articles)) + 
  geom_bar(stat = "identity",fill="#69b3a2", color="#e9ecef", alpha=0.9)



### Explore article categories
# First we create a data frame containing Unique categories and get the count for each category
unique_category = as.data.frame(table(news_frame$category))

# Then we sort the categories in decreasing order based on the article count
unique_category = unique_category[order(-unique_category$Freq),]

# Rename the column names
unique_category = unique_category %>%
  rename(
    "Category" = "Var1",
    "Articles" = "Freq"
  )

#write_xlsx(unique_category,glue(dirname(rstudioapi::getSourceEditorContext()$path),"/unique_catgegory.xlsx"))

# Reset row names/index
rownames(unique_category) <- NULL                 

# Print category with most category counts
head(unique_category,10)

# Print category with least category counts
tail(unique_category,10)

# We remove the categories with count below 100
unique_category = filter(unique_category, Articles>100)
news_frame <- filter(news_frame, category %in% unique_category$Category)
#We have 26 unique categories but it is highly imbalanced. 

# Lets get some stats
unique_category %>% 
  # Get Mean and Median category size
  summarise(mean_category_size   = mean(Articles),
            median_category_size = median(Articles))

# Plot categories
unique_category %>%
     mutate(Category = fct_reorder(Category, Articles)) %>%
     ggplot(aes(y = Category, x = Articles)) + 
     geom_bar(stat = "identity",fill="#69b3a2", color="#e9ecef", alpha=0.9) +
     xlab("")
# As we can see its a rather imbalanced class distribution

# Thus for training we'll downsample the data to the by taking only 1004 latest artciles from all categories
# For testing we'll use the entire test category


#######################################################################################
# Explore headlines
# Define a function nwords that takes string as input and provides count of words as output

nwords = function(string, pseudo=F)                  
{
  ifelse( pseudo,                   # F = False (default), T = True 
          pattern <- "\\S+",        # If pseudo is F, pattern = Non space characters including numbers   
          pattern <- "[[:alpha:]]+" # else, pattern = only alphabetic characters considered 
  )
  str_count(string, pattern)
}

# Lets create a count column for headline word count
news_frame  = news_frame %>%                   
  mutate(headline_word_count = as.numeric(lapply(news_frame$headline, nwords)))
# We used "lapply" to apply function to entire column (This will take some time)
# Then we used "as.numeric" to convert it to numeric

# Remove articles with less than a word count of 0
news_frame = filter(news_frame, headline_word_count > 0)

# Lets print some characteristics
news_frame %>% 
  # Describe headline word count
  summarise(min    = min(headline_word_count),
            max    = max(headline_word_count),
            mean   = mean(headline_word_count),
            median = median(headline_word_count),
            standard_deviation = sd(headline_word_count))

# Visualise word count

news_frame$headline_word_count = as.numeric(as.character(news_frame$headline_word_count))

# Histogram with density plot
ggplot(news_frame, aes(x=headline_word_count)) +                   
  geom_histogram(binwidth=1, fill="#69b3a2", color="#e9ecef", alpha=0.9) +          
  geom_vline(aes(xintercept=mean(headline_word_count)), color="blue",linetype="dashed") + 
  xlim(0, 30)

# Cheking normality by plotting Q-Q plot
ggplot(news_frame, aes(sample = headline_word_count)) + 
  stat_qq(color="#69b3a2") + stat_qq_line(color="blue", linetype="dashed")

#Post cleaning it doesn't look so normal[distributed]
#######################################################################################

############### Text pre-processing

# Create a copy of healine to do the processing on and save the original headlines
news_frame  = news_frame %>%                   
  mutate(headline_text = news_frame$headline)

##### Step 1 - Convert to lower case

# Use tolower to convert to lower case
news_frame$headline_text = tolower(news_frame$headline_text)
# Print sample to observe
head(news_frame$headline_text)

##### Step 2 - Remove puncuations

# We can use Character replace function and  pass regex to remove all punctuations.
# The gsub() function always deals with regular expressions.

news_frame$headline_text = gsub('[[:punct:] ]+',' ',news_frame$headline_text)

# Print sample to observe
head(news_frame$headline_text)

##### Step 3 - Remove stopwords

# List standard English stop words
stopwords("en")

# Apply removeWords to provide stopwords as arg to remove the stopwords from all strings in the column
news_frame$headline_text = removeWords(news_frame$headline_text, 
                                       stopwords("en"))

# Print sample to observe
head(news_frame$headline_text)

##### Step 4 - Lemmatisation

# Apply lemmatize_strings 
news_frame$headline_text = lemmatize_strings(news_frame$headline_text)

# Print sample to observe
head(news_frame$headline_text)
#########################################################################################

# Lets re-create a count column for headline word count after pre-processing
news_frame  = news_frame %>%                   
  mutate(headline_word_count = as.numeric(lapply(news_frame$headline_text, nwords)))
# We used "lapply" to apply function to entire column (This will take some time)
# Then we used "as.numeric" to convert it to numeric

# Remove articles with less than a word count of 3
news_frame = filter(news_frame, headline_word_count > 3)

# Lets print some characteristics
news_frame %>% 
  # Describe headline word count
  summarise(min    = min(headline_word_count),
            max    = max(headline_word_count),
            mean   = mean(headline_word_count),
            median = median(headline_word_count),
            standard_deviation = sd(headline_word_count))

# Visualise word count

news_frame$headline_word_count = as.numeric(as.character(news_frame$headline_word_count))

# Histogram with density plot
ggplot(news_frame, aes(x=headline_word_count)) +                   
  geom_histogram(binwidth=1, fill="#69b3a2", color="#e9ecef", alpha=0.9) +          
  geom_vline(aes(xintercept=mean(headline_word_count)), color="blue",linetype="dashed") + 
  xlim(0, 30)

# Cheking normality by plotting Q-Q plot
ggplot(news_frame, aes(sample = headline_word_count)) + 
  stat_qq(color="#69b3a2") + stat_qq_line(color="blue", linetype="dashed")

#Post cleaning it doesn't look as much normal[distributed] but not that far off

########################################################################################
########## Prepare for modelling


all_data = select(news_frame,c(category,headline_text)) %>%
  mutate(category = factor(category)) #%>%
#mutate(enc_cat = factor(transform(lab_enc,category)))
#Split the data set.
#First we need to convert 'category' to a factor variable. \
#  Then we can split the data into training and testing datasets using initial_split() from rsample."

data_split    = initial_split(data = all_data, strata = category, prop = .8)

train_set     = training(data_split) 
test_set      = testing(data_split)

#Next we need to preprocess the data in preparation for modeling. \
#  Currently we have 'headline_text' data, and we need to construct numeric, quantitative features for machine learning based on that text. \
#  As before, we can use recipes to construct the set of preprocessing steps we want to perform.

train_rec     = recipe(category ~ headline_text, data = train_set)

#Now we add steps to process the text. We use textrecipes to handle the text variable. \
#  Step 1 tokenize the text to words with step_tokenize(). 
#    By default this uses tokenizers::tokenize_words(). 
#  Step 2 We use step_tokenfilter() to only keep the 500 most frequent tokens, to avoid creating too many variables in our first model
#  Step 3 As the some categories have much mroe articles which leads the data to be imbalanced data. So we down sample big sets to a level similar to others
#  Step 4 we use step_tfidf() to compute tf-idf."

train_rec     = train_rec %>%
  step_tokenize(headline_text) %>%
  step_tokenfilter(headline_text, max_tokens = 500) %>%
  step_downsample(category) %>%
  step_tfidf(headline_text)

# Creating a function to collect metrics
model_metrics = function(model_fit, test_set, label_col, model_name) 
{
  # Predict classes
  class_predictions = predict(model_fit, test_set)
  # Predict probability 
  prob_predictions  = tryCatch({
    predict(model_fit, test_set, type = "prob")
  },
  error=function(cond){
    return (NULL)
  })
  # Get accuracy
  met_accuracy      = bind_cols(test_set,class_predictions) %>% 
    accuracy(truth = label_col, estimate = .pred_class)
  # Get precision
  met_precision     = bind_cols(test_set,class_predictions) %>% 
    precision(truth = label_col, estimate = .pred_class)
  # Get recall
  met_recall        = bind_cols(test_set,class_predictions) %>% 
    recall(truth = label_col, estimate = .pred_class)
  # Get ROC
  met_roc           = tryCatch({
    bind_cols(test_set,prob_predictions) %>% 
      roc_aunp(label_col, names(prob_predictions)) %>% 
      mutate(Model=model_name)
  },
  error=function(cond){
    return (data.frame(.estimate = c(0)))
  })
  # Return a frame with all metrics
  return (c(model_name,met_accuracy$.estimate,
                      met_precision$.estimate,
                      met_recall$.estimate,
                      met_roc$.estimate))
}
#############################################
# NULL model
# Get start time
null_start_time = Sys.time()
# Create model
null_spec       = null_model() %>%
  set_mode("classification") %>%
  set_engine("parsnip")

# Build pipeline
null_wf         = workflow() %>%
  add_recipe(train_rec) %>%
  add_model(null_spec)
# Fit the model 
null_fit        = fit(null_wf,data = train_set)
# Fetch metrics
metrics_null    = model_metrics(null_fit, test_set, test_set$category, "Null model")
# Get process end time
null_end_time = Sys.time()
# Get time taken
null_time = null_end_time - null_start_time

# Get null model predictions
null_predictions = predict(null_fit, test_set)

# Plot confusion matrix as a heatmap
bind_cols(test_set,null_predictions) %>% 
  conf_mat(truth = category, estimate = .pred_class) %>%
  autoplot(type = "heatmap") + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
###########################################################################
#########################################################################################################################
## Naive Bayes approach (Probabilistic Model)
#  Get start time
nb_start_time = Sys.time()
# Create model
nb_spec       = naive_Bayes() %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

# Build pipeline
nb_wf         = workflow() %>%
  add_recipe(train_rec) %>%
  add_model(nb_spec)
# Fit the model 
nb_fit        = fit(nb_wf,data = train_set)
# Fetch metrics
metrics_nb    = model_metrics(nb_fit, test_set, test_set$category, "Naive Bayes")
# Get process end time
nb_end_time = Sys.time()
# Get process time taken
nb_time = nb_end_time - nb_start_time

# Get null model predictions
nb_predictions = predict(nb_fit, test_set)

# Plot confusion matrix as a heatmap
bind_cols(test_set,nb_predictions) %>% 
  conf_mat(truth = category, estimate = .pred_class) %>%
  autoplot(type = "heatmap") + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
#########################################################################################################################
#########################################################################################################################
# KNN 
#  Get start time
knn_start_time = Sys.time()
# Create model
knn_spec       = nearest_neighbor(neighbors = 5, weight_func = "optimal") %>%
                  set_mode("classification") %>%
                  set_engine("kknn")
# Build pipeline
knn_wf         = workflow() %>%
                  add_recipe(train_rec) %>%
                  add_model(knn_spec)
# Fit the model 
knn_fit        = fit(knn_wf,data = train_set)
# Fetch metrics
metrics_knn    = model_metrics(knn_fit, test_set, test_set$category, "K nearest neighbour")
# Get process end time
knn_end_time = Sys.time()
# Get process time taken
knn_time = knn_end_time - knn_start_time
#########################################################################################################################
#########################################################################################################################
## Decision trees

#  Get start time
tree_start_time = Sys.time()
# Create model
tree_spec      =  decision_tree() %>%
                    set_mode("classification") %>%
                    set_engine("C5.0")
# Build pipeline
tree_wf        = workflow() %>%  
                    add_recipe(train_rec) %>%
                    add_model(tree_spec)
# Fit the model 
tree_fit             = fit(tree_wf,data = train_set)
# Fetch metrics
metrics_tree    = model_metrics(tree_fit, test_set, test_set$category, "Decision tree")
# Get process end time
tree_end_time = Sys.time()
# Get process time taken
tree_time = tree_end_time - tree_start_time
#########################################################################################################################
#########################################################################################################################
# Random trees

#  Get start time
rf_cls_start_time = Sys.time()
# Create model
rf_cls_spec    = rand_forest(trees = 200, min_n = 5) %>% 
                    # This model can be used for classification or regression, so set mode
                    set_mode("classification") %>% 
                    set_engine("randomForest")
# Build pipeline
rf_cls_wf      = workflow() %>%  
                    add_recipe(train_rec) %>%
                    add_model(rf_cls_spec)
# Fit the model 
rf_cls_fit     = fit(rf_cls_wf,data = train_set)
# Fetch metrics
metrics_rf_cls = model_metrics(rf_cls_fit, test_set, test_set$category, "Random Forest")
# Get process end time
rf_cls_end_time = Sys.time()
# Get process time taken
rf_cls_time = rf_cls_end_time - rf_cls_start_time
# Get null model predictions
rf_cls_predictions = predict(rf_cls_fit, test_set)
# Plot confusion matrix as a heatmap
bind_cols(test_set,rf_cls_predictions) %>% 
  conf_mat(truth = category, estimate = .pred_class) %>%
  autoplot(type = "heatmap") + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
#########################################################################################################################
#########################################################################################################################
# SVM Linear

#  Get start time
svm_cls_start_time = Sys.time()
# Create model
svm_cls_spec    = svm_linear(cost = 1) %>% 
                    # This model can be used for classification or regression, so set mode
                    set_mode("classification") %>% 
                    set_engine("LiblineaR")
# Build pipeline
svm_cls_wf      = workflow() %>%  
                    add_recipe(train_rec) %>%
                    add_model(svm_cls_spec)
# Fit the model 
svm_cls_fit     = fit(svm_cls_wf,data = train_set)
# Fetch metrics
metrics_svm_cls = model_metrics(svm_cls_fit, test_set, test_set$category, "SVM Linear")
# Get process end time
svm_cls_end_time = Sys.time()
# Get process time taken
svm_cls_time = svm_cls_end_time - svm_cls_start_time
#########################################################################################################################
#########################################################################################################################
# SVM Polynomial

#  Get start time
svm_poly_start_time = Sys.time()
# Create model
svm_poly_cls_spec = svm_poly(cost = 1) %>% 
                      # This model can be used for classification or regression, so set mode
                      set_mode("classification") %>% 
                      set_engine("kernlab")
# Build pipeline
svm_poly_cls_wf   = workflow() %>%  
                      add_recipe(train_rec) %>%
                      add_model(svm_poly_cls_spec)
# Fit the model
svm_poly_cls_fit     = fit(svm_poly_cls_wf,data = train_set)
# Fetch metrics
metrics_svm_poly_cls = model_metrics(svm_poly_cls_fit, test_set, test_set$category, "SVM Polynomial")
# Get process end time
svm_poly_end_time = Sys.time()
# Get process time taken
svm_poly_time = svm_poly_end_time - svm_poly_start_time
#########################################################################################################################
#########################################################################################################################
# XG Boost trees

#  Get start time
xgboost_start_time = Sys.time()
# Create model
xgboost_cls_spec    = boost_tree(trees = 200) %>% 
                      set_mode("classification") %>% 
                      set_engine("xgboost") 
# Build pipeline
xgboost_cls_wf      = workflow() %>%  
                    add_recipe(train_rec) %>%
                    add_model(xgboost_cls_spec)

# Fit the model
xgboost_cls_fit     = fit(xgboost_cls_wf,data = train_set)
# Fetch metrics
metrics_xgboost_cls = model_metrics(xgboost_cls_fit, test_set, test_set$category, "XG Boost")
# Get process end time
xgboost_end_time = Sys.time()
# Get process time taken
xgboost_time = xgboost_end_time - xgboost_start_time
#########################################################################################################################
#########################################################################################################################
# Multinomial logistic regression - keras

#  Get start time
mr_kr_start_time = Sys.time()
# Create model
mr_kr_cls_spec    = multinom_reg(penalty = 0.1) %>% 
                      set_mode("classification") %>% 
                      set_engine("keras")
# Build pipeline
mr_kr_cls_wf      = workflow() %>%  
                      add_recipe(train_rec) %>%
                      add_model(mr_kr_cls_spec)

# Fit the model
mr_kr_cls_fit     = fit(mr_kr_cls_wf,data = train_set)
# Fetch metrics
metrics_mr_kr_cls = tryCatch({
                              model_metrics(mr_kr_cls_fit, test_set, 
                                            test_set$category, "Multinominal LR")
                            },
                            error=function(cond){
                              return (c("Multinominal LR", 0,0,0,0))})
# Get process end time
mr_kr_end_time = Sys.time()
# Get process time taken
mr_kr_time = mr_kr_end_time - mr_kr_start_time
#########################################################################################################################
#########################################################################################################################
# Create a metric data frame
metric_frame0 = data.frame(metrics_null, metrics_nb, metrics_knn, 
                           metrics_tree, metrics_rf_cls, 
                          metrics_xgboost_cls, metrics_svm_cls,
                          metrics_svm_poly_cls, metrics_mr_kr_cls)
# Transpose the frame
metric_frame = data.table(t(metric_frame0))
# Set column names
names(metric_frame) = c("Model", "Accuracy", "Precision", 
                        "Recall", "ROC_AUC")
# Get F1 score and insert process time
metric_frame = metric_frame %>%
          mutate(Precision = as.numeric(Precision)) %>%
          mutate(Recall = as.numeric(Recall)) %>%
          mutate(F1Score = 2*(Precision * Recall)/(Precision + Recall)) %>%
          mutate(ProcessTime = c(null_time, nb_time, 
                                 knn_time, tree_time,
                                 rf_cls_time, xgboost_time,
                                 svm_cls_time, svm_poly_time,
                                 mr_kr_time)) %>%
          mutate(ProcessTime = round(ProcessTime,2))
        
metric_frame
# Get end time for the entire code 
end_time <- Sys.time()
# Get total run time 
total_run_time = end_time - start_time
# Write to an excel file the metrics data
#write_xlsx(metric_frame,glue(dirname(rstudioapi::getSourceEditorContext()$path),"/metric_frame.xlsx"))

#########################################################################################################################
#########################################################################################################################

# Plot accuracy
metric_frame %>%
  mutate(Model = fct_reorder(Model, Accuracy)) %>%
  ggplot(aes(y = Accuracy, x = Model)) + 
  geom_bar(stat = "identity",fill="#69b3a2", color="#e9ecef", alpha=0.9) +
  xlab("")

# Plot F1Score
metric_frame %>%
  mutate(Model = fct_reorder(Model, F1Score)) %>%
  ggplot(aes(y = F1Score, x = Model)) + 
  geom_bar(stat = "identity",fill="#69b3a2", color="#e9ecef", alpha=0.9) +
  xlab("")
#########################################################################################################################
#########################################################################################################################
