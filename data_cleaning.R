##Load libraries
library(tidyverse)

set.seed(1)

##Read in data
train <- read_csv(file = "data/train.csv") %>% as_tibble()
test <- read_csv(file = "data/test.csv") %>% as_tibble()

##Handle missing values
colSums(is.na(train))
colSums(is.na(test))

# Drop missing data
train <- train %>%
  select(-Cabin) %>% # drop Cabin because there are so many na
  na.omit()

# Split train into train and validation because 
# the test dataset doesn't provide survival results
train_set <- train %>% sample_frac(0.7)
train_val <- train %>% setdiff(train_set)

# Check if we have na's left
colSums(is.na(train))
colSums(is.na(train_set))
colSums(is.na(train_val))
colSums(is.na(test))
