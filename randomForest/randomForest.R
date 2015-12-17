# This example will construct a basic decision tree using the rpart package
# to predict whether an individual's income is greater or less than 50k USD 
# based on 14 observable predictors
setwd("D:/R Data/UCI Machine Learning Repository") # NOTE: Change your working directory
# as appropriate here before running!      

library(caret)

# Download adult income data
url.train <- "http://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
url.names <- "http://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names"
download.file(url.train, destfile = "HousingValues.csv")
download.file(url.names, destfile = "HousingFeatureNames.txt")

# Read the training and test data into memory
set.seed(123)
train <- read.table("HousingValues.csv")


# Feature names can be found by referring to the downloaded names .txt file above
names(train) <- c("CRIM",      #per capita crime rate by town
                  "ZN",        #proportion of residential land zoned for lots over 25,000 sq.ft.
                  "INDUS",     #proportion of non-retail business acres per town
                  "CHAS",      #Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
                  "NOX",       #nitric oxides concentration (parts per 10 million)
                  "RM",        #average number of rooms per dwelling
                  "AGE",       #proportion of owner-occupied units built prior to 1940
                  "DIS",       #weighted distances to five Boston employment centres
                  "RAD",       #index of accessibility to radial highways
                  "TAX",       #full-value property-tax rate per $10,000
                  "PTRATIO",   #pupil-teacher ratio by town
                  "B",         #1000(Bk - 0.63)^2 where Bk is the proportion of African Americans by town
                  "LSTAT",     #% lower status of the population
                  "MEDV")      #Median value of owner-occupied homes in $1000's
      
inTrain <- createDataPartition(train$MEDV, p = .75, list = FALSE)
train <- train[inTrain,]
test <- train[-inTrain,]

# Use caret package to implement random forest model
mtry_def <- floor(sqrt(ncol(train))*.75) # How many columns to select in each bootstrap sample?
t_grid <- expand.grid(mtry= c(mtry_def))

# Train model
set.seed(1234)
model.rf <- train(MEDV ~ .,
                  data = train,
                  method = "rf",
                  ntree = 50, # How many trees to grow in total?
                  tuneGrid = t_grid)
print(model.rf)

# How did our model do?
predictions <- predict(model.rf, test[,1:13])
RMSE <- sqrt(sum((predictions - test$MEDV)^2)/length(predictions))
print(RMSE)
#How to interpret this RMSE?
print(RMSE/mean(test$MEDV)) # RMSE is about 9% as large as the mean of our outcome, not bad!

# What if we include more columns in our bootstramp sample?
# This time, we will not specify tuneGrid so caret will automatically try different levels of mtry
set.seed(1234)
model.rf <- train(MEDV ~ .,
                  data = train,
                  method = "rf",
                  ntree = 50)
                  
print(model.rf)

# How did our model do?
predictions <- predict(model.rf, test[,1:13])
RMSE <- sqrt(sum((predictions - test$MEDV)^2)/length(predictions))
print(RMSE)
#How to interpret this RMSE?
print(RMSE/mean(test$MEDV)) # Instead of 2 columns, use 7 columns and error is about 7% of the mean; nice improvement

# Effect of increasing number of trees? Instead of growing 50 trees, we will grow 500
set.seed(1234)
model.rf <- train(MEDV ~ .,
                  data = train,
                  method = "rf",
                  ntree = 500)
print(model.rf)

# How did our model do?
predictions <- predict(model.rf, test[,1:13])
RMSE <- sqrt(sum((predictions - test$MEDV)^2)/length(predictions))
print(RMSE)
#How to interpret this RMSE?
print(RMSE/mean(test$MEDV)) # RMSE reduced to about 5.9% as large as the mean of our outcome
                            # But notice the extra computation time that was required! randomForest
                            # can be very computationally intensive with lots of big trees and large
                            # data set.
plot(model.rf$finalModel)
# Looks like after about 300 trees, not much improvement in error level