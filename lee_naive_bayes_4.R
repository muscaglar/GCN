library("caret")library(pscl)library('glmnet')library("klaR")library("h2o")library("magrittr")library("dplyr")
library("pillar")library(mlr)library('FSelector')

#### Data Prep
data <- read.csv('/Desktop/known_all_factors.csv',header=T,na.strings=c(""))
data[data$gcn_present == 0,]$gcn_present <- "No"
data[data$gcn_present == 1,]$gcn_present <- "Yes"
data$gcn_present <- as.factor(data$gcn_present)
'%ni%' <- Negate('%in%')  # define 'not in' func
options(scipen=999)  # prevents printing scientific notations.
set.seed(890)
trainDataIndex <- createDataPartition(data$gcn_present, p=0.9, list = F)  # 80% training data
train <- data[trainDataIndex, ]
test <- data[-trainDataIndex, ]
#### Down train data
table(train$gcn_present)
set.seed(100)
down_train <- downSample(x = train[, colnames(train) %ni% "gcn_present"],y = train$gcn_present)
colnames(down_train)[colnames(down_train)=="Class"] <- "gcn_present"
table(down_train$gcn_present)
detach("package:caret", unload=TRUE)
train_task = makeClassifTask(data = down_train, target = "gcn_present")
test_task = makeClassifTask(data = test, target = "gcn_present")

#### Feature importance
im_feat <- generateFilterValuesData(train_task)#, method = c("FSelectorRcpp_information.gain","FSelector_chi.squared"))
plotFilterValues(im_feat,n.show = 20)
#### Drop Features
train_task <- dropFeatures(task = train_task,features = c("SI9_oldham","density_score","pond_density","SI9_lee"))
test_task <- dropFeatures(task = test_task,features = c("SI9_oldham","density_score","pond_density","SI9_lee"))
#### Normalise
train_task <- normalizeFeatures(train_task,method = "standardize")
test_task <- normalizeFeatures(test_task,method = "standardize")

#### NaiveBayes
selected_model = makeLearner("classif.naiveBayes")
NB_mlr = train(selected_model, train_task)
NB_mlr$learner.model
#### Predict
predictions_mlr = as.data.frame(predict(NB_mlr, test_task))
#### Result
calculateConfusionMatrix(predict(NB_mlr, test_task), relative = FALSE, sums = FALSE,set = "both")

#### Logistic
logistic.learner <- makeLearner("classif.logreg",predict.type = "response")
cv.logistic <- crossval(learner = logistic.learner,task = train_task,iters = 3,stratify = TRUE,measures = acc,show.info = F)
#### Accuracy
cv.logistic$aggr
cv.logistic$measures.test
fmodel <- train(logistic.learner,train_task)
getLearnerModel(fmodel)
#### Predict
fpmodel = as.data.frame(predict(fmodel, test_task))
#### Result
calculateConfusionMatrix(predict(fmodel, test_task), relative = FALSE, sums = FALSE,set = "both")

#### Regression Trees
library("rpart")
getParamSet("classif.rpart")
makeatree <- makeLearner("classif.rpart", predict.type = "response")
set_cv <- makeResampleDesc("CV",iters = 3L)
#Search for hyperparameters
gs <- makeParamSet(
  makeIntegerParam("minsplit",lower = 10, upper = 50),
  makeIntegerParam("minbucket", lower = 5, upper = 50),
  makeNumericParam("cp", lower = 0.001, upper = 0.2)
)
#### Grid Search
gscontrol <- makeTuneControlGrid()
#### Tuning
stune <- tuneParams(learner = makeatree, resampling = set_cv, task = train_task, par.set = gs, control = gscontrol, measures = acc)
stune$x
stune$y
#### Apply Tuning
t.tree <- setHyperPars(makeatree, par.vals = stune$x)
#### Training
t.rpart <- train(t.tree, train_task)
getLearnerModel(t.rpart)
#### Predict
tpmodel <- predict(t.rpart, test_task)
#### Result
calculateConfusionMatrix(tpmodel, relative = FALSE, sums = FALSE,set = "both")

#### Random Forest
getParamSet("classif.randomForest")
#### Train
rf <- makeLearner("classif.randomForest", predict.type = "response", par.vals = list(ntree = 200, mtry = 3))
rf$par.vals <- list(
  importance = TRUE
)
#### Tune
rf_param <- makeParamSet(
  makeIntegerParam("ntree",lower = 50, upper = 500),
  makeIntegerParam("mtry", lower = 3, upper = 10),
  makeIntegerParam("nodesize", lower = 10, upper = 50)
)
rancontrol <- makeTuneControlRandom(maxit = 50L)
#### Validate
set_cv <- makeResampleDesc("CV",iters = 3L)
#### Tune
rf_tune <- tuneParams(learner = rf, resampling = set_cv, task = train_task, par.set = rf_param, control = rancontrol, measures = acc)
#### Apply Tuning
rf.tree <- setHyperPars(rf, par.vals = rf_tune$x)
#### Train
rforest <- train(rf.tree, train_task)
getLearnerModel(t.rpart)
#### Predict
rfmodel <- predict(rforest, test_task)
#### Result
calculateConfusionMatrix(rfmodel, relative = FALSE, sums = FALSE,set = "both")

#### Support Vector Machine
library('kernlab')
getParamSet("classif.ksvm")
ksvm <- makeLearner("classif.ksvm", predict.type = "response")
pssvm <- makeParamSet(
  makeDiscreteParam("C", values = 2^c(-8,-4,-2,0)), #cost parameters
  makeDiscreteParam("sigma", values = 2^c(-8,-4,0,4)) #RBF Kernel Parameter
)
ctrl <- makeTuneControlGrid()
#### Tunning
res <- tuneParams(ksvm, task = train_task, resampling = set_cv, par.set = pssvm, control = ctrl,measures = acc)
res$y
acc.test.mean 
#### Apply Tuning
t.svm <- setHyperPars(ksvm, par.vals = res$x)
#### Train
par.svm <- train(ksvm, train_task)
#### Predict
predict.svm <- predict(par.svm, test_task)
#### Result
calculateConfusionMatrix(predict.svm, relative = FALSE, sums = FALSE,set = "both")


