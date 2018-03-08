##Load libraries
library(tidyverse)
library(modelr)
library(readr)

##Read in data
train <- read_csv(file = "data/train.csv") %>% as_tibble()
test <- read_csv(file = "data/test.csv") %>% as_tibble()

##Drop unwanted variables
train <- train %>% select(-PassengerId, -Name, -Ticket, -Cabin)

##Drop missing values
colSums(is.na(train))
train <- na.omit(train)
test$Age[is.na(test$Age)] <- median(na.omit(test$Age))
test$Fare[is.na(test$Fare)] <- median(na.omit(test$Fare))

##Split train into training and validation set
set.seed(1)
training <- train %>% sample_frac(0.7)
validation <- setdiff(train, training)


##Logistic regression
glm_fit <- glm(Survived ~., data = train, family = binomial)
summary(glm_fit)
coef(glm_fit)
summary(glm_fit)$coef

glm_probs <- predict(glm_fit, type = "response")
glm_probs[1:10]

train <- train %>%
  mutate(probs = glm_probs) %>%
  mutate(probs = ifelse(probs > 0.5, 1, 0))

attach(train)
table(probs, Survived)
mean(probs == Survived)

##10-fold cross validation
train_10fold <- train %>% crossv_kfold(k = 10, id = "fold")

mod <- function(df) {
  glm(Survived ~., data = df, family = binomial)
}

train_10fold <- train_10fold %>%
  mutate(mod = map(train, mod)) 



##KNN
###Data preparation
training_knn <- training
validation_knn <- validation
training_knn <- training_knn %>%
  mutate(Sex = ifelse(Sex == "male", 1, 0),
         Embarked = sapply(Embarked, switch, "C" = 0, "Q" = 1, "S" = 2))
validation_knn <- validation_knn %>%
  mutate(Sex = ifelse(Sex == "male", 1, 0),
         Embarked = sapply(Embarked, switch, "C" = 0, "Q" = 1, "S" = 2))

###Modeling
library(class)
train_X <- training_knn %>% select(-Survived)
test_X <- validation_knn %>% select(-Survived)
train_survival <- training_knn %>% select(Survived) 

set.seed(1)
knn_pred <- knn(train_X, test_X, t(train_survival), k = 3)
table(knn_pred, validation_knn$Survived)
mean(knn_pred == validation_knn$Survived)

##LDA 
library(MASS)
lda_fit <- lda(Survived ~., data = training)
lda_fit
plot(lda_fit)

lda_predict <- predict(lda_fit, validation)
names(lda_predict)
lda_class <- lda_predict$class
table(lda_class, validation_knn$Survived)
mean(lda_class == validation_knn$Survived)

lda_preds <- predict(lda_fit, test)$class
lda_test <- test %>% 
  mutate(Survived = lda_preds) %>%
  dplyr::select(PassengerId, Survived)

write_csv(lda_test, "lda_test.csv")

##QDA
qda_fit <- qda(Survived ~., data = training)
qda_fit
plot(qda_fit)

qda_predict <- predict(qda_fit, validation)
names(qda_predict)
qda_class <- qda_predict$class
table(qda_class, validation_knn$Survived)
mean(qda_class == validation_knn$Survived)

