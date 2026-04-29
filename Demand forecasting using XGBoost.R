install.packages(c("readr", "dplyr", "ggplot2", "xgboost", "caret"))

# Load libraries
library(readr)
library(dplyr)
library(ggplot2)
library(xgboost)
library(caret)

# 1. Load dataset
data <- read_csv("npd_example.csv")

# 2. Study dataset
head(data)
str(data)
summary(data)
colSums(is.na(data))

# 3. Split into training and testing sets
set.seed(123)

train_index <- createDataPartition(data$demand, p = 0.8, list = FALSE)

train_data <- data[train_index, ]
test_data  <- data[-train_index, ]

# 4. Prepare data for XGBoost

# Separate target variable
x_train <- train_data %>% select(-demand)
y_train <- train_data$demand

x_test <- test_data %>% select(-demand)
y_test <- test_data$demand

# Convert categorical variables into dummy variables
dummy_model <- dummyVars(~ ., data = x_train)

x_train_matrix <- predict(dummy_model, newdata = x_train)
x_test_matrix  <- predict(dummy_model, newdata = x_test)

# Convert to xgboost format
dtrain <- xgb.DMatrix(data = x_train_matrix, label = y_train)
dtest  <- xgb.DMatrix(data = x_test_matrix, label = y_test)

# 5. Fit XGBoost model
xgb_model <- xgboost(
  x = x_train_matrix,
  y = y_train,
  objective = "reg:squarederror",
  nrounds = 100,
  max_depth = 4,
  learning_rate = 0.1
)

# 6. Predict demand on test set
predictions <- predict(xgb_model, x_test_matrix)

results <- data.frame(
  Actual = y_test,
  Predicted = predictions
)

head(results)

# 7. Measure accuracy using RMSE
rmse <- sqrt(mean((results$Actual - results$Predicted)^2))
rmse

# 8. Plot forecast demand against actual values
ggplot(results, aes(x = Actual, y = Predicted)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0) +
  labs(
    title = "Actual vs Predicted Demand",
    x = "Actual Demand",
    y = "Predicted Demand"
  )

# Optional: plot actual and predicted values by observation number
results$Index <- 1:nrow(results)

ggplot(results, aes(x = Index)) +
  geom_line(aes(y = Actual)) +
  geom_line(aes(y = Predicted)) +
  labs(
    title = "Actual Demand vs Forecast Demand",
    x = "Observation",
    y = "Demand"
  )
