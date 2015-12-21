library(caret)

# Download adult income data
url.train <- "http://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
url.names <- "http://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names"
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

# Let's compare output from OLS, ridge, lasso, least angle regression (lars), and elasticnet which linearly 
# combines the L1 and L2 penalties of the lasso and ridge methods.
model.names <- c("OLS",
                 "LassoTop5",
                 "LassoAllVars",
                 "Ridge",
                 "ElasticNetTop5",
                 "ElasticNetAllVars")
set.seed(12928)
model.ols <- train(MEDV ~ .,
                   data = train,
                   method = "lm")
coefficients <- data.frame(model.ols$finalModel$coefficients[-1])

model.lasso <- train(MEDV ~ .,
                     data = train,
                     method = "lasso",
                     tuneLength = 20)
lasso.coef <- predict(model.lasso$finalModel, type='coef', mode='norm')$coefficients
coefficients <- cbind(coefficients,
                      lasso.coef[6,],
                      lasso.coef[nrow(lasso.coef),])

model.ridge <- train(MEDV ~ .,
                     data = train,
                     method = "ridge",
                     tuneLength = 30)
ridge.coef <- predict(model.ridge$finalModel, type='coef', mode='norm')$coefficients
coefficients <- cbind(coefficients,
                      ridge.coef[nrow(ridge.coef),])

model.enet <- train(MEDV ~ .,
                    data = train,
                    method = "enet",
                    tuneLength = 8)
enet.coef <- predict(model.enet$finalModel, type='coef', mode='norm')$coefficients
coefficients <- cbind(coefficients,
                      enet.coef[6,],
                      enet.coef[nrow(enet.coef),])

names(coefficients) <- model.names
print(coefficients)

print(model.lasso$finalModel) #INDUS gets added in the 10th step and then removed in the 13th step and readded at the end
print(lasso.coef) # See it happen here

print(model.ridge$finalModel) #Only 14 steps in ridge, one for each predictor

plot(model.ridge)
plot(model.lasso)
plot(model.enet) # RMSE was used to select the optimal model using  the smallest value.
                 # The final values used for the model were fraction = 1 and lambda = 0.01.
                 # Notice enet uses both of the tuning parameters from lasso and ridge

# Principal Component Regression
# Dimensionality Reduction technique
model.pcr <- train(MEDV ~ .,
                   data = train,
                   method = "pcr",
                   tuneLength = 13)
plot(model.pcr) # Looks like 11 components minimizes RMSE