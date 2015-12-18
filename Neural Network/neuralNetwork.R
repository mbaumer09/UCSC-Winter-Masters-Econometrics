# This example will construct a basic decision tree using the rpart package
# to predict whether an individual's income is greater or less than 50k USD 
# based on 14 observable predictors

library(caret)
library(nnet)
library(NeuralNetTools)

url.train <- "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
url.test <- "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
url.names <- "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names"
download.file(url.train, destfile = "adult_train.csv")
download.file(url.test, destfile = "adult_test.csv")
download.file(url.names, destfile = "adult_names.txt")

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
file.remove("adult_train.csv")
file.remove("adult_test.csv")

# Use caret package to train a model using neural net on all vars
set.seed(1414)
start <- proc.time()[3]
model.nn <- train(IncomeLevel ~ .,
                  data = train,
                  method = "nnet")
print(model.nn)
predictions <- predict(model.nn, test[,1:14])
accuracy <- sum(predictions == test[,15])/length(test[,15])
print(accuracy)
end <- proc.time()[3]
print(paste("This took ", round(end-start, digits = 1), " seconds", sep = ""))

# Use feature selection procedure from example
# We will try a different model this time, linear discriminant analysis

set.seed(1414)
model.lda <- train(IncomeLevel ~ .,
                   data = train,
                   method = "lda")
plot(varImp(model.lda))

keeps <- c("EducationNum",
           "Relationship",
           "Age",
           "HoursPerWeek",
           "MaritalStatus",
           "IncomeLevel")

train.reduced <- train[,which(names(train) %in% keeps)]
test.reduced <- test[,which(names(test) %in% keeps)]
set.seed(1414)
start <- proc.time()[3]
model.nn <- train(IncomeLevel ~ .,
                  data = train.reduced,
                  method = "nnet")

print(model.nn)

predictions <- predict(model.nn, test.reduced[,1:5])
accuracy <- sum(predictions == test.reduced[,6])/length(test.reduced[,6])
print(accuracy)
end <- proc.time()[3]
print(paste("This took ", round(end-start, digits = 1), " seconds", sep=""))

# For visualization purposes, lets take only columns which are non-factors (or binary)

keeps <- c("EducationNum",
           "Age",
           "HoursPerWeek",
           "Sex",
           "CapitalGain",
           "IncomeLevel")

train.reduced <- train[,which(names(train) %in% keeps)]
test.reduced <- test[,which(names(test) %in% keeps)]
set.seed(1414)
start <- proc.time()[3]
model.nn <- train(IncomeLevel ~ .,
                  data = train.reduced,
                  method = "nnet")

print(model.nn)
predictions <- predict(model.nn, test.reduced[,1:5])
accuracy <- sum(predictions == test.reduced[,6])/length(test.reduced[,6])
print(accuracy)
end <- proc.time()[3]
print(round(end-start, digits = 1))

# Use NeuralNetTools package to visualize this
plotnet(model.nn$finalModel)

# Bonus: another feature importance methodology! 
garson(model.nn$finalModel)
