# This example will construct a basic decision tree using the rpart package
# to predict whether an individual's income is greater or less than 50k USD 
# based on 14 observable predictors

setwd("C:/Users/USER/Dropbox/Side Project/Machine Learning/Alan Examples/Decision Tree") 
library(caret)
library(ElemStatLearn)

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
inTrain <- createDataPartition(train$IncomeLevel, p = .5, list = FALSE)
train <- train[inTrain,]
validation <- train[-inTrain,]

# Let's use our lessons from feature selection
model.tree  <- train(IncomeLevel ~ .,
                   data = train,
                   method = "rpart")
plot(varImp(model.tree), top = 10)

# k Nearest Neighbors
start <- proc.time()[3]
model.knn <- train(IncomeLevel ~ .,
                   data = train,
                   method = "knn")
print(model.knn)
validation.predictions <- predict(model.knn, validation[,1:5])
validation.acc <- sum(validation.predictions == validation[,6])/length(validation[,6])
print(paste("The model correctly predicted the validation set outcome ", accuracy*100, "% of the time", sep=""))
predictions <- predict(model.knn, test[,1:5])
accuracy <- sum(predictions == test[,6])/length(test[,6])
print(paste("The model correctly predicted the test outcome ", round(accuracy*100,digits=2), "% of the time", sep=""))
end <- proc.time()[3]
print(paste("This took ", round(end-start, digits = 2), " seconds", sep=""))

#Visualization
# Taken from ?mixture.example from ElemStatLearn package
# Let's build a knn model using only Age and Education level variables
data(mixture.example)
par(mfrow = c(2,3))
x <- mixture.example$x
g <- mixture.example$y
x.mod <- lm( g ~ x)
# Figure 2.1:
plot(x, col=ifelse(g==1,"red", "green"), xlab="x1", ylab="x2", main = "OLS")
abline( (0.5-coef(x.mod)[1])/coef(x.mod)[3], -coef(x.mod)[2]/coef(x.mod)[3])
ghat <- ifelse( fitted(x.mod)>0.5, 1, 0)

xnew <- mixture.example$xnew

library(class)
mod1 <- knn(x[,1:2], xnew[,1:2], g, k=1, prob=TRUE)
prob <- attr(mod1, "prob")
prob <- ifelse( mod1=="1", prob, 1-prob) # prob is voting fraction for winning class!
# Now it is voting fraction for red==1

px1 <- mixture.example$px1
px2 <- mixture.example$px2

prob1 <- matrix(prob, length(px1), length(px2))
contour(px1, px2, prob1, levels=0.5, labels="", xlab="x1", ylab="x2", main=
              "1-nearest neighbour")
# adding the points to the plot:
points(x, col=ifelse(g==1, "red", "green"))

mod3 <- knn(x, xnew, g, k=3, prob=TRUE)
prob <- attr(mod3, "prob")
prob <- ifelse( mod3=="1", prob, 1-prob) # prob is voting fraction for winning class!
# Now it is voting fraction for red==1

px1 <- mixture.example$px1
px2 <- mixture.example$px2

prob3 <- matrix(prob, length(px1), length(px2))
contour(px1, px2, prob3, levels=0.5, labels="", xlab="x1", ylab="x2", main=
              "3-nearest neighbour")
# adding the points to the plot:
points(x, col=ifelse(g==1, "red", "green"))

mod15 <- knn(x, xnew, g, k=10, prob=TRUE)


prob <- attr(mod15, "prob")
prob <- ifelse( mod15=="1", prob, 1-prob) # prob is voting fraction for winning class!
# Now it is voting fraction for red==1

px1 <- mixture.example$px1
px2 <- mixture.example$px2

prob15 <- matrix(prob, length(px1), length(px2))
contour(px1, px2, prob15, levels=0.5, labels="", xlab="x1", ylab="x2", main=
              "10-nearest neighbour")
# adding the points to the plot:
points(x, col=ifelse(g==1, "red", "green"))

mod20 <- knn(x, xnew, g, k=25, prob=TRUE)


prob <- attr(mod20, "prob")
prob <- ifelse( mod20=="1", prob, 1-prob) # prob is voting fraction for winning class!
# Now it is voting fraction for red==1

px1 <- mixture.example$px1
px2 <- mixture.example$px2

prob20 <- matrix(prob, length(px1), length(px2))
contour(px1, px2, prob20, levels=0.5, labels="", xlab="x1", ylab="x2", main=
              "25-nearest neighbour")
# adding the points to the plot:
points(x, col=ifelse(g==1, "red", "green"))

mod50 <- knn(x, xnew, g, k=50, prob=TRUE)


prob <- attr(mod50, "prob")
prob <- ifelse( mod50=="1", prob, 1-prob) # prob is voting fraction for winning class!
# Now it is voting fraction for red==1

px1 <- mixture.example$px1
px2 <- mixture.example$px2

prob50 <- matrix(prob, length(px1), length(px2))
contour(px1, px2, prob50, levels=0.5, labels="", xlab="x1", ylab="x2", main=
              "50-nearest neighbour")
# adding the points to the plot:
points(x, col=ifelse(g==1, "red", "green"))
