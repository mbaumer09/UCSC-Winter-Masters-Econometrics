# This example will construct a basic decision tree using the rpart package
# to predict whether an individual's income is greater or less than 50k USD 
# based on 14 observable predictors

setwd("C:/Users/USER/Dropbox/Side Project/Machine Learning/Alan Examples/Decision Tree") 
library(caret)

# Download adult income data
#url.train <- "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
#url.test <- "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
#url.names <- "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names"
#download.file(url.train, destfile = "adult_train.csv")
#download.file(url.test, destfile = "adult_test.csv")
#download.file(url.names, destfile = "adult_names.txt")

# Read the training and test data into memory
train <- read.csv("adult_train.csv", header = FALSE)

# The test data has an unnecessary first line that messes stuff up, this fixes that problem
all_content <- readLines("adult_test.csv")
skip_first <- all_content[-1]
test <- read.csv(textConnection(skip_first), header = FALSE)

# The data file doesn't have the column names in its header, add those in manually...
varNames <- c("Age", 
              "WorkClass",
              "fnlwgt",
              "Education",
              "EducationNum",
              "MaritalStatus",
              "Occupation",
              "Relationship",
              "Race",
              "Sex",
              "CapitalGain",
              "CapitalLoss",
              "HoursPerWeek",
              "NativeCountry",
              "IncomeLevel")

names(train) <- varNames
names(test) <- varNames
levels(test$IncomeLevel) <- levels(train$IncomeLevel)

# Use feature selection procedure from example
# One slight hitch in this plan: this is not a regression problem, it's classification
model.lda <- train(IncomeLevel ~ .,
                  data = train,
                  method = "lda")
print(varImp(model.rf))

# Neural Network package nnet
start <- proc.time()[3]
model.nn <- train(IncomeLevel ~ .,
                     data = train,
                     method = "nnet")
print(model.nn)
predictions <- predict(model.nn, test[,1:14])
accuracy <- sum(predictions == test[,15])/length(test[,15])
print(accuracy)
end <- proc.time()[3]
print(round(end-start, digits = 1))