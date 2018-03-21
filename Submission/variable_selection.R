##Load libraries
library(tidyverse)
library(modelr)
library(readr)
library(reshape2)
library(leaps)

# Set Seed
set.seed(1)

##Read in data
train <- read_csv(file = "data/train.csv") %>% as_tibble() %>% select(-Name, -PassengerId, -Ticket)
test <- read_csv(file = "data/test.csv") %>% as_tibble() %>% select(-Name, -PassengerId, -Ticket)

##Handle missing values
colSums(is.na(train))
colSums(is.na(test))

# Drop missing data
train <- train %>%
  select(-Cabin) %>% # drop Cabin because there are so many na
  na.omit()
test <- test %>%
  select(-Cabin) %>% # drop Cabin because there are so many na
  na.omit()

# Split train into train and validation because 
# the test dataset doesn't provide survival results
training <- train %>% sample_frac(0.8)
validation <- setdiff(train, training)

# Check if we have na's left
colSums(is.na(train))
colSums(is.na(training))
colSums(is.na(validation))
colSums(is.na(test))

##Data preparation
training <- training %>%
  mutate(Sex = ifelse(Sex == "male", 1, 0),
         Embarked = sapply(Embarked, switch, "C" = 0, "Q" = 1, "S" = 2))

validation <- validation %>%
  mutate(Sex = ifelse(Sex == "male", 1, 0),
         Embarked = sapply(Embarked, switch, "C" = 0, "Q" = 1, "S" = 2))

test <- test%>%
  mutate(Sex = ifelse(Sex == "male", 1, 0),
         Embarked = sapply(Embarked, switch, "C" = 0, "Q" = 1, "S" = 2))


# Best Subset Selection
best_fits <- regsubsets(Survived ~ ., data = training, nvmax = 10)
best_fits_results <- summary(best_fits)

best_fit_data <- tibble(
  num_pred = 1:7,
  RSS = best_fits_results$rss,
  R2 = best_fits_results$rsq,
  Adj_R2 = best_fits_results$adjr2,
  Cp = best_fits_results$cp,
  BIC = best_fits_results$bic
) %>%
  gather(key = "statistic", value= "value", - num_pred)


# Plot the graph for each RSS, Adj_R2, Cp, and BIC
best_fit_data %>%
  filter(statistic != "R2") %>%
  ggplot(aes(x = num_pred, y = value)) +
  geom_point() +
  geom_line() +
  facet_wrap(~ statistic, ncol = 2, scales = "free")

# Identify min number of predictors (CP, BIC)
best_fit_data %>%
  group_by(statistic) %>%
  filter(value == min(value), statistic %in% c("Cp", "BIC"))

# Identify min number of predictors (Adj_R2)
best_fit_data %>%
  group_by(statistic) %>%
  filter(value == max(value), statistic == "Adj_R2")

# Best model when using 5 predictors
best_fits %>% coef(5)


# Forward and backward model selection
# Forward stepwise selection
fwd_fit <- regsubsets(Survived ~ ., 
                      data = training, 
                      nvmax = 7, 
                      method = "forward")

# Backward stepwise selection
bwd_fit <- regsubsets(Survived ~ ., 
                      data = training, 
                      nvmax = 7, 
                      method = "backward")

# Get the results
fwd_results <- summary(fwd_fit)
bwd_results <- summary(bwd_fit)

# Define a helper function
regsub_plot <- function(df_regsubset)
{ # get results data
  df <- summary(df_regsubset)
  
  # reshape data
  best_fit_data <- tibble(
    num_pred = 1:length(df$rss),
    RSS = df$rss,
    Rsqr = df$rsq,
    Adjr2 = df$adjr2,
    Cp = df$cp,
    BIC = df$bic
  ) %>%
    gather(key = "statistic", value = "value", -num_pred)
  
  # make the plot
  best_fit_data %>%
    filter(statistic != "R2") %>%
    ggplot(aes(x = num_pred, y = value)) +
    geom_point() +
    geom_line() +
    facet_wrap(~ statistic, ncol = 2, scales = "free")
}

# Plot the results
# Forward
regsub_plot(fwd_fit)

# Backward
regsub_plot(bwd_fit)

# Extract coefficients for model with 5 preds
coef(best_fits, 5)
coef(fwd_fit, 5)
coef(bwd_fit, 5)

# It is pretty clear that we should use Pclass, Sex, Age, SibSp, and Embarked
# We will not used Parch and Fare as predictors