# When you have a potentially huge number of predictor variables, how do you know which ones to pick?
# The problem is overfitting: if you include more predictors, your training set error will necessarily
# always decrease, but this does not mean that your out of sample error is decreasing with it!
# One easy answer to to select a method that naturally selects them! 
# For example, lasso, rpart, and randomForest
# But what if you're using a different model?

setwd("D:/R Data/UCI Machine Learning Repository") # NOTE: Change your working directory
                                                   # as appropriate here before running!      
library(caret)

# Download adult income data
#url.train <- "http://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
#url.names <- "http://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names"
#download.file(url.train, destfile = "HousingValues.csv")
#download.file(url.names, destfile = "HousingFeatureNames.txt")

# Read the training and test data into memory
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
                  "AA",        #1000(AA - 0.63)^2 where AA is the proportion of African Americans by town
                  "LSTAT",     #% lower status of the population
                  "MEDV")      #Median value of owner-occupied homes in $1000's

# Let's start nice and easy with a no-frills OLS linear model
model.lm <- train(MEDV ~ .,
                  data = train,
                  method = "lm")
# Compare results to what you get from using lm()
model <- lm(MEDV ~ .,
            data = train)
# The caret package has a canned method of determine relative importance of different features
# using an ROC curve
plot(varImp(model.lm))


# Now let's figure out what to include
# Including multiple highly correlated features does not offer very much explanatory power beyond
# the power of including just one, but it negatively impacts variance of estimates so let's use that as our
# first pass
correlations <- cor(train[,1:13])
print(correlations)
highCorrelations <- findCorrelation(correlations, cutoff = .75, verbose = TRUE)

# It would seem that features 3,5, and 10 have high correlations with other things 
# and thus we should remove them. What happens to our variable importance picture now?
train <- train[,-highCorrelations]
model.lm <- train(MEDV ~ .,
                  data = train,
                  method = "lm")
plot(varImp(model.lm))

# The caret package has another nice prepackaged methodology for feature selection: rfeControl()
# which uses a method called Recursive Feature Elimination

set.seed(134)
# Define the control using a random forest selection function
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
results <- rfe(train[,1:10], train[,11], sizes=c(1:10), rfeControl=control)
# summarize the results
print(results)
# list the chosen features
predictors(results)
# plot the results
plot(results, type=c("g", "o"))

# Lowest RMSE at 9 predictors but this is only slightly better than what was attained with 6, so might
# still be better to only include 6 for final model. It also suggests the top 5 out of those 9.

# Lasso for feature selection

train <- read.table("HousingValues.csv")

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
                  "AA",        #1000(AA - 0.63)^2 where AA is the proportion of African Americans by town
                  "LSTAT",     #% lower status of the population
                  "MEDV")      #Median value of owner-occupied homes in $1000's
train <- train[,-9]
model.lasso <- train(MEDV ~ .,
                     data = train,
                     method = "lasso")
print(model.lasso$finalModel)
plot(model.lasso$finalModel, xvar = "penalty")

# This plot shows which variables get set to zero in the lasso as lambda (the penalty parameter) is increased.
# Which variables get set to zero last? How do these compare to what we found using the first two methods?

