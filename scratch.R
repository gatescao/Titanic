##Load libraries
library(tidyverse)
library(modelr)
library(readr)
library(reshape2)

##Read in data
train <- read_csv(file = "data/train.csv") %>% as_tibble()
test <- read_csv(file = "data/test.csv") %>% as_tibble()

##Drop unwanted variables
train <- train %>% dplyr::select(-PassengerId, -Name, -Ticket, -Cabin)

##Handling missing values
colSums(is.na(train))
train$Age[is.na(train$Age)] <- median(na.omit(train$Age))
train$Fare[is.na(train$Fare)] <- median(na.omit(train$Fare))
train$Embarked[is.na(train$Embarked)] <- "S"

colSums(is.na(test))
test$Age[is.na(test$Age)] <- median(na.omit(test$Age))
test$Fare[is.na(test$Fare)] <- median(na.omit(test$Fare))


##Split train into training and validation set
set.seed(1)
training <- train %>% sample_frac(0.8)
validation <- setdiff(train, training)

##EDA
train %>% group_by(Pclass) %>% summarize(survival_rate = mean(Survived)) %>%
  ggplot(aes(Pclass, survival_rate)) +
  geom_col()

train %>% group_by(Sex) %>% summarize(survival_rate = mean(Survived)) %>%
  ggplot(aes(Sex, survival_rate)) +
  geom_col()

train %>% group_by(Embarked) %>% summarize(survival_rate = mean(Survived)) %>%
  ggplot(aes(Embarked, survival_rate)) +
  geom_col()

train %>% group_by(SibSp) %>% summarize(survival_rate = mean(Survived)) %>%
  ggplot(aes(SibSp, survival_rate)) +
  geom_col()

train %>% group_by(Parch) %>% summarize(survival_rate = mean(Survived)) %>%
  ggplot(aes(Parch, survival_rate)) +
  geom_col()

cormat <- train %>%
  dplyr::select(Survived, Pclass, Age, Fare, Parch) %>%
  cor()

melted_cormat <- melt(cormat)
ggplot(data = melted_cormat, aes(x = Var1, y = Var2, fill = value)) + 
  geom_tile()



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

##Logistic regression
glm_fit <- glm(Survived ~., data = training, family = binomial)

glm_probs <- predict(glm_fit, training, type = "response")

training <- training %>%
    mutate(probs = glm_probs) %>%
    mutate(probs = ifelse(probs > 0.5, 1, 0))

###Training accuracy
attach(training)
table(probs, Survived)
mean(probs == Survived)

###Testing accuracy
glm_probs <- predict(glm_fit, validation, type = "response")
validation <- validation %>%
    mutate(probs = glm_probs) %>%
    mutate(probs = ifelse(probs > 0.5, 1, 0))

attach(validation)
table(probs, Survived)
mean(probs == Survived)  #0.77

glm_preds <- predict(glm_fit, test, type = "response")
glm_test <- test %>% 
  mutate(Survived = ifelse(glm_preds > 0.5, 1, 0)) %>%
  dplyr::select(PassengerId, Survived)

write_csv(glm_test, "glm_test.csv")


##10-fold cross validation
train_10fold <- training %>% crossv_kfold(k = 10, id = "fold")

mod <- function(df) {
  glm(Survived ~., data = df, family = binomial)
}

train_10fold <- train_10fold %>%
  mutate(mod = map(train, mod)) 


##Classification trees
library(tree)
library(rattle)
training$Survived <- as.factor(training$Survived)
tree_fit <- tree(Survived ~., data = training)
summary(tree_fit)

tree_preds <- predict(tree_fit, validation, type = "class")
table(tree_preds, validation$Survived)
mean(tree_preds == validation$Survived)

set.seed(1)
cv_survived <- cv.tree(tree_fit, FUN = prune.misclass)
cv_survived

prune_survival <- prune.misclass(tree_fit, best = 7)
tree_preds <- predict(prune, validation, type="class")
table(tree_preds, validation$Survived)
mean(tree_preds == validation$Survived)

tree_preds <- predict(tree_fit, test, type = "class")
tree_test <- test %>% 
  mutate(Survived = tree_preds) %>%
  dplyr::select(PassengerId, Survived)

write_csv(tree_test, "tree_test.csv")



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

