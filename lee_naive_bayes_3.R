####### Naive Bayes 3   http://uc-r.github.io/naive_bayes#caret

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
training <- data[trainDataIndex, ]
testing <- data[-trainDataIndex, ]

table(training$gcn_present)
set.seed(100)
down_train <- downSample(x = train[, colnames(train) %ni% "gcn_present"],y = train$gcn_present)
colnames(down_train)[colnames(down_train)=="Class"] <- "gcn_present"
table(down_train$gcn_present)

h2o.no_progress()
h2o.init()

x <- setdiff(names(down_train), y)
y <- "gcn_present"

x.test <- setdiff(names(test), y)
y.test <- "gcn_present"

train.h2o <- down_train %>% mutate_if(is.factor, factor, ordered = FALSE) %>% as.h2o()
test.h2o <- testing %>% mutate_if(is.factor, factor, ordered = FALSE) %>% as.h2o()

nb.h2o <- h2o.naiveBayes(
  x = x,
  y = y,
  training_frame = train.h2o,
  nfolds = 10,
  laplace = 0
)

h2o.confusionMatrix(nb.h2o)
test_pp    <- predict(nb.h2o, newdata = test.h2o)
#test_pp.h2o <- test_pp %>%
 # mutate_if(is.factor, factor, ordered = FALSE) %>%
  #as.h2o()

h2o.performance(nb.h2o, newdata = test.h2o )
h2o.predict(nb.h2o, newdata = test.h2o)
h2o.shutdown(prompt = FALSE)