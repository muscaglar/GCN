####### Naive Bayes 2   http://uc-r.github.io/naive_bayes#caret

library("caret")
library(pscl)
library('glmnet')
library("klaR")
library("h2o")
library("magrittr")
library("dplyr")
library("pillar")
library(mlr)

data <- read.csv('/Volumes/MusDrive/Dropbox/ZSHARE_Mus_Lee/R/known_all_factors.csv',header=T,na.strings=c(""))
data$gcn_present <- as.factor(data$gcn_present)

'%ni%' <- Negate('%in%')  # define 'not in' func
options(scipen=999)  # prevents printing scientific notations.

set.seed(890)
trainDataIndex <- createDataPartition(data$gcn_present, p=0.90, list = F)  # 85% training data
train <- data[trainDataIndex, ]
test <- data[-trainDataIndex, ]

table(train$gcn_present)
set.seed(100)
down_train <- downSample(x = train[, colnames(train) %ni% "gcn_present"],y = train$gcn_present)
colnames(down_train)[colnames(down_train)=="Class"] <- "gcn_present"
table(down_train$gcn_present)

features <- setdiff(names(train), "gcn_present")
x <- train[, features]
y <- train$gcn_present

train_control <- trainControl(
  method = "cv", 
  number = 10
)

nb.m1 <- train(
  x = x,
  y = y,
  method = "nb",
  trControl = train_control
)

confusionMatrix(nb.m1)

pred <- predict(nb.m1, newdata = test)
confusionMatrix(pred, test$gcn_present)