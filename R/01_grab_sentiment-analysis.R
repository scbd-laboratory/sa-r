# install Library
install.packages(c('readr', 'tm', 'wordcloud', 'e1071', 'gmodels'))

# import library
library(readr)
library(tm)
library(wordcloud)
library(e1071)
library(gmodels)

# Import Data
grab_labelled <- read.csv("data-raw/01_grab-training.csv")
grab_labelled$sentiment <- factor(grab_labelled$sentiment)

# Check the counts of positive and negative scores
table(grab_labelled$sentiment)

# Create a corpus from the sentences
grab_corpus <- VCorpus(VectorSource(grab_labelled$text))

# create a document-term sparse matrix directly from the corpus
grab_dtm <- DocumentTermMatrix(grab_corpus)

# creating training and test datasets
grab_dtm_train <- grab_dtm[1:25, ]
grab_dtm_test  <- grab_dtm[26:35, ]

# also save the labels
grab_train_labels <- grab_labelled[1:25, ]$sentiment
grab_test_labels  <- grab_labelled[26:35, ]$sentiment

# check that the proportion  is similar
prop.table(table(grab_train_labels))

prop.table(table(grab_test_labels))

rm(grab_dtm_train)
rm(grab_dtm_test)
rm(grab_train_labels)
rm(grab_test_labels)

# Create random samples
set.seed(123)
train_index <- sample(35, 25)

grab_train <- grab_labelled[train_index, ]
grab_test  <- grab_labelled[-train_index, ]

# check the proportion of class variable
prop.table(table(grab_train$sentiment))
prop.table(table(grab_test$sentiment))

train_corpus <- VCorpus(VectorSource(grab_train$text))
test_corpus <- VCorpus(VectorSource(grab_test$text))

# subset the training data into spam and ham groups
positive <- subset(grab_train, sentiment == 1)
negative  <- subset(grab_train, sentiment == 0)

wordcloud(positive$text, max.words = 40, scale = c(3, 0.5))
wordcloud(negative$text, max.words = 40, scale = c(3, 0.5))


# create a document-term sparse matrix directly for train and test
train_dtm <- DocumentTermMatrix(train_corpus)

test_dtm <- DocumentTermMatrix(test_corpus)

train_dtm
test_dtm


# create function to convert counts to a factor
convert_counts <- function(x)
  {x <- ifelse(x > 0, "Yes", "No")}

# apply() convert_counts() to columns of train/test data
train_dtm_binary <- apply(train_dtm, MARGIN = 2, convert_counts)
test_dtm_binary  <- apply(test_dtm, MARGIN = 2, convert_counts)

# Create the model
grab_classifier <- naiveBayes(as.matrix(train_dtm_binary), grab_train$sentiment)          

# Apply model to predict test dataset
grab_test_pred <- predict(grab_classifier, as.matrix(test_dtm_binary))
head(grab_test_pred)

# Confussion Matrix
CrossTable(grab_test_pred, grab_test$sentiment,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))


# Import Prediction Data
pred_labelled <- read.csv("data-raw/01_grab-predict.csv")

# Create a corpus from the sentences
pred_corpus <- VCorpus(VectorSource(pred_labelled$text))

# create a document-term sparse matrix directly from the corpus
pred_dtm <- DocumentTermMatrix(pred_corpus)

# Crate a binary matrix
pred_dtm_binary  <- apply(pred_dtm, MARGIN = 2, convert_counts)

# Apply Prediction 
grab_predict_pred <- predict(grab_classifier, as.matrix(pred_dtm_binary))
head(grab_predict_pred)
