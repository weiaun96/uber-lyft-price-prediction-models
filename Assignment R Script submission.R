###### Section 1 -  Reading Data & Data Exploration ######

#Before running the following code block use "Session" tab to "Set Working Directory" to "Source File Location"
filelocation<-getwd()
setwd(filelocation)
cab_rides <- read.csv('cab_rides.csv', header = T)

#install.packages('DataExplorer')
library(DataExplorer)
library(dplyr)

## Examine structure of the dataset and summary statistics
head(rides) #Display top few rows of dataframe rides
head(rides, n=10) # first 10 recorrides2
dim(rides) # dimension
str(rides) # structure
plot_str(rides, fontSize = 40) #Visualise dataset structure
plot_str(weather, fontSize = 40)
hist(rides$distance)
summary (rides)

#Basic Statistic
config <- configure_report(
  global_ggtheme = quote(theme_minimal(base_size = 14))
)

create_report(rides, output_file = "statistics by cab type", y="cab_type", config = config)

# check for missing data in the data frame
is.na(rides)
sum (is.na(rides))
colSums(sapply(rides,is.na))

# % of missing values in each variable
plot_missing(rides)
plot_histogram(rides)
plot_density(rides)

#plotting boxplot
plot_boxplot(rides, 'surge_multiplier')
boxplot(rides)

#plotting scatterplot
plot_scatterplot(rides, 'surge_multiplier')

#plotting correlation for discrete categories
plot_correlation(rides,'surge_multiplier')

create_report(rides)




###### Section 3 -  Data Preparation ######
### Missing Value Treatment ###
install.packages('VIM')
library(VIM)
install.packages('mice')
library(mice)
#install.packages('missForest') # Here this package is used to introduce x% of missing values (NA) into the dataset
#library(missForest) #missForest is not used as it takes a long iteration time

# Visualizing missing values using VIM package
vim_plot <- aggr(rides, numbers=TRUE, prop = c(TRUE, FALSE))

# To convert empty spaces to missing values
rides2 <- rides #keep rides as original, only do imputation to the copy
View (rides2)
rides2 <- mutate_all(rides2,na_if,"")
plot_missing(rides2)
colSums(sapply(rides2,is.na))
hist(rides2$price)

# Imputing missing values in continuous variables using mean
rides2$price = ifelse(is.na(rides2$price),
                      ave(rides2$price, FUN = function(x) mean(x, na.rm = TRUE)),
                      rides2$price)
colSums(sapply(rides2,is.na))

# Imputing missing values using mice - Method 1

imputed_rides2 <- mice(rides2, m=3)
rides3 <- complete (imputed_rides2)
View (rides3)

### Remove Irrelevant Columns ###
#column id & product_id, which is unique, irrelevant to the learning, therefore is removed from the table
cab_rides <- cab_rides[,c(-8,-9)]
#since name is the type of ride which will also affect the price, a new column is added by doing label encoding to the name
factors<-factor(cab_rides$name)
factors
cab_rides$nameno<-as.numeric(factors)
cab_rides$cab_type<-as.factor(cab_rides$cab_type)
cab_rides$destination<-as.factor(cab_rides$destination)
cab_rides$source<-as.factor(cab_rides$source)
cab_rides$name<-as.factor(cab_rides$name)

#since target variable is the price, it is put as first column
cab_rides <- cab_rides[, c(6, 1, 7, 9, 2, 3, 4, 5, 8)]

#due to dataset too big, split function is used to reduce the dataset
library(caTools)
set.seed(123)
split11 <- sample.split(cab_rides, SplitRatio = 0.2)
cab_ride_split = subset(cab_rides, split11 == TRUE)
rides = subset(cab_ride_split, split11 == TRUE)
dim(cab_rides)
dim(cab_ride_split)
dim(rides)

### Feature Selection ###
# ensure results are repeatable
set.seed(7)
# load the library
library(mlbench)
library(caret)

# prepare training scheme
control <- trainControl(method="repeatedcv", number=10, repeats=3)
# train the model
model <- train(price~ distance+surge_multiplier+nameno+time_stamp+destination+source, data=rides2, method="glmStepAIC", preProcess="scale", trControl=control)
# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)

### Train-Test Split ###
library(caTools)
set.seed(123)
split = sample.split(rides2$price, SplitRatio = 0.6)
training_set = subset(rides2, split == TRUE)
test_set = subset(rides2, split == FALSE)


###### Section 4 -  Model Building ######
### Linear Regression ###
regressor1 = lm(formula = price ~ ., data = training_set)
summary(regressor1)

#install.packages("jtools")
library(jtools)
#library(Rcpp)
summ(regressor1, confint = TRUE)

# Predicting the Test set results
y_pred = predict(regressor1, newdata = test_set)
y_pred
summary(y_pred)
table(y_pred, test_set$price) # Comparing the predicted and actual value
cbind(y_pred, test_set$price)

library(caret)
RMSE(y_pred, test_set$price) # Root mean squared error
MAE(y_pred, test_set$price) # Mean Absolute Error

# Building the optimal model using Backward Elimination
regressor2 = lm(formula = price ~ nameno + distance + surge_multiplier,
                data = training_set)
summary(regressor2)
y_pred2 = predict(regressor2, newdata = test_set)
summ(regressor2, confint = TRUE, ci.width=.5)

regressor3 = lm(formula = price ~ nameno + distance,
                data = training_set)
summary(regressor3)
y_pred3 = predict(regressor3, newdata = test_set)
summ(regressor3, confint = TRUE, ci.width=.5)

regressor4 = lm(formula = price ~ nameno,
                data = rides2)
summary(regressor4)
y_pred4 = predict(regressor4, newdata = test_set)
summ(regressor4, confint = TRUE, ci.width=.5)


RMSE(y_pred2, test_set$price) # Root mean squared error
MAE(y_pred2, test_set$price) # Mean Absolute Error
RMSE(y_pred3, test_set$price) # Root mean squared error
MAE(y_pred3, test_set$price) # Mean Absolute Error
RMSE(y_pred4, test_set$price) # Root mean squared error
MAE(y_pred4, test_set$price) # Mean Absolute Error



### Ridge Regression ###
# The data is converted into matrix because the model "glmnet" needs data in matrix form
x <- data.matrix(training_set[,2:9])
y <- training_set$price

x_test <- data.matrix(test_set[,2:9])
y_test <- test_set$price

# Ridge regression - glmnet parameter alpha=0 for ridge regression
# For numerical prediction choose family - gaussian
# glmnet by defaut chooses 100 lambda values that are data dependent
l_ridge <- glmnet(x, y, family="gaussian", alpha=0)
plot(l_ridge, xvar = 'lambda', label=T)
summary(l_ridge)

#finding best value for lambda
cv_out_ridge = cv.glmnet(x, y, alpha =0)
plot (cv_out_ridge)
names(cv_out_ridge) # outputs created by cv_out_ridge

# two lambda values may be noted. 'lambda.min', 'lambda.1se'- lambda for error within 1 standard deviation
lambda_min <- cv_out_ridge$lambda.min
lambda_min
lambda_1se<- cv_out_ridge$lambda.1se
lambda_1se

#Plot the ridge regression output once again
plot(l_ridge, xvar = 'lambda', label=T)
abline(v = log(cv_out_ridge$lambda.1se), col = "red", lty = "dashed")
abline(v = log(cv_out_ridge$lambda.min), col = "blue", lty = "dashed")

# Next, set lambda to one of these values and build the model
l_ridge_final <- glmnet(x, y, family="gaussian", lambda = lambda_1se, alpha=0)
coef(l_ridge_final)
plot(coef(l_ridge_final))

# Prediction with training set
p1 <- predict(l_ridge_final, x) 
rmse_l_ridge_final <- sqrt(mean((training_set$price)^2))
rmse_l_ridge_final
library(caret)
RMSE(p1, test_set$price) # Root mean squared error
MAE(p1, test_set$price)

### Lasso Regression ###
cv_out_lasso = cv.glmnet(x, y, alpha = 1)
plot (cv_out_lasso)
names(cv_out_lasso)
# two lambda values may be noted. 'lambda.min', 'lambda.1se'- lambda for error within 1 standard deviation
lambda_min <- cv_out_lasso$lambda.min
lambda_min
lambda_1se<- cv_out_lasso$lambda.1se
lambda_1se

# Plot the lasso regression output once again
l_lasso <- glmnet(x, y, family="gaussian", alpha=1)
plot(l_lasso, xvar = 'lambda', label=T)
abline(v = log(cv_out_lasso$lambda.1se), col = "red", lty = "dashed")
abline(v = log(cv_out_lasso$lambda.min), col = "blue", lty = "dashed")

# Next, set lambda to one of these values and build the model
l_lasso_final <- glmnet(x, y, family="gaussian", lambda = lambda_1se, alpha=0)
coef(l_lasso_final)
plot(coef(l_lasso_final))

# Prediction with training set
p2 <- predict(l_lasso_final, x) #remember to use x and not train_data
rmse_l_lasso_final <- sqrt(mean((train_data$price)^2))
rmse_l_lasso_final

#~~~~~ Ridge Regression Evaluation ~~~~~#
# Compute R^2 from true and predicted values
eval_results <- function(true, predicted, df) {
  SSE <- sum((predicted - true)^2)
  SST <- sum((true - mean(true))^2)
  R_square <- 1 - SSE / SST
  RMSE = sqrt(SSE/nrow(df))
  
  
  # Model performance metrics
  data.frame(
    RMSE = RMSE,
    Rsquare = R_square
  )
  
}

# Prediction and evaluation on train data
predictions_train <- predict(l_ridge, s = lambda_min, newx = x)
eval_results(y, predictions_train, training_set)

# Prediction and evaluation on test data
predictions_test <- predict(l_ridge, s = lambda_min, newx = x_test)
eval_results(y_test, predictions_test, test_set)

#~~~~~ Lasso Regression Evaluation ~~~~~#
lasso_model <- glmnet(x, y, alpha = 1, lambda = lambda_min, standardize = TRUE)


# Prediction and evaluation on train data
predictions_train <- predict(l_lasso, s = lambda_min, newx = x)
eval_results(y, predictions_train, training_set)

# Prediction and evaluation on test data
predictions_test <- predict(l_lasso, s = lambda_min, newx = x_test)
eval_results(y_test, predictions_test, test_set)


##### Net Elastic Regression #####
# Set training control
train_cont <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 5,
                           search = "random",
                           verboseIter = TRUE)

# Train the model
elastic_reg <- train(price ~ .,
                     data = training_set,
                     method = "glmnet",
                     preProcess = c("center", "scale"),
                     tuneLength = 10,
                     trControl = train_cont)

elastic_reg <- train(price ~ nameno+distance+time_stamp+surge_multiplier,
                     data = training_set,
                     method = "glmnet",
                     preProcess = c("center", "scale"),
                     tuneLength = 10,
                     trControl = train_cont)

# Best tuning parameter
elastic_reg$bestTune

# Make predictions on training set
predictions_train <- predict(elastic_reg, x)
eval_results(y, predictions_train, training_set) 

# Make predictions on test set
predictions_test <- predict(elastic_reg, x_test)
eval_results(y_test, predictions_test, test_set)

# ~~~~~~~~~~~~~~~~~~~~  Default SVM Model using the RBF kernel ~~~~~~~~~~~~~~~~~~~~~
svm_rbf <- svm(price~., data = training_set)
summary(svm_rbf)

pred = predict (svm_rbf, test_set)
pred
#table(pred, test_set$price)

library(caret)
summary(pred)
RMSE(pred, test_set$price) # Root mean squared error
MAE(pred, test_set$price) # Mean Absolute Error

svm_rbf2 <- svm(price~., data = training_set, cost=4, gamma=0.5)
summary(svm_rbf2)

pred2 = predict (svm_rbf2, test_set)
pred2
#table(pred, test_set$price)

library(caret)
summary(pred)
RMSE(pred2, test_set$price) # Root mean squared error
MAE(pred2, test_set$price) # Mean Absolute Error



# ~~~~~~~~~~~~~~~~~~~ SVM Parameter tuning ~~~~~~~~~~~~~~~~~~~ 
obj <- tune(svm, price~., data = training_set,
            ranges = list(gamma = 2^(-1:1), cost = 2^(2:4)),
            tunecontrol = tune.control(sampling = "fix")
)
summary(obj)
plot(obj)