##Load libraries
library(tidyverse)

##Read in data
train <- read_csv(file = "data/train.csv") %>% as_tibble()
test <- read_csv(file = "data/test.csv") %>% as_tibble()

##Handle missing values
colSums(is.na(train))
train <- train %>% select(-PassengerId, -Name, -Ticket, -Cabin)

train$Age[is.na(train$Age)] <- median(na.omit(train$Age))

##Logistic regression
glm_fit <- glm(Survived ~., data = train, family = binomial)