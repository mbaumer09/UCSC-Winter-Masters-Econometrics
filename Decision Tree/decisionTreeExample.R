# This example will construct a basic decision tree using the rpart package
# to predict whether an individual's income is greater or less than 50k USD 
# based on 14 observable predictors

setwd("C:/Users/USER/Dropbox/Side Project/Machine Learning/Alan Examples/Decision Tree") 
library(rpart)
library(rpart2)
library(rpart.plot)
library(rattle) # Rattle package may as you to install something else; after this is installed
                # you must restart R for it to work!
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
# Use rpart to grow our tree
tree <- rpart(IncomeLevel ~ .,
              data = train,
              method = "class")
print(tree)
plotcp(tree)

# Turn that model into a picture
fancyRpartPlot(tree, main = "Adult Income Level") # Notice how the tree only utilized 3 predictor variables
                                                  # out of the 14 available!

# Now use our decision tree model to predict on the test data set to see 
# how accurate it is

outcomes <- predict(tree, test[1:14])
predictions <- ifelse(outcomes[,1] >= .5, " <=50K", " >50K") # if our tree assigns probability of at least .5
                                                             # to outcome " <=50K" then that's its prediction

accuracy <- round(sum(predictions == test[,15])/length(predictions), digits = 4)
print(paste("The model correctly predicted the test outcome ", accuracy*100, "% of the time", sep=""))

# Let's do this using the caret package
model.tree <- train(IncomeLevel ~ .,
                    data = train,
                    method = "rpart2")
fancyRpartPlot(model.tree$finalModel)

predictions <- predict(model.tree, test[1:14])
accuracy <- round(sum(predictions == test[,15])/length(predictions), digits = 4)
print(paste("The model correctly predicted the test outcome ", accuracy*100, "% of the time", sep=""))
