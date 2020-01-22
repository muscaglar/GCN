######## Data Prep

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

#data$gcn_present <- as.factor(data$gcn_present)

'%ni%' <- Negate('%in%')  # define 'not in' func
options(scipen=999)  # prevents printing scientific notations.

set.seed(890)
trainDataIndex <- createDataPartition(data$gcn_present, p=0.8, list = F)  # 80% training data
train <- data[trainDataIndex, ]
test <- data[-trainDataIndex, ]

######## Logit - Hawes recommended : https://www.machinelearningplus.com/machine-learning/logistic-regression-tutorial-examples-r/

#I do have class imbalance, but more concerning, where gcn_present = FALSE,this is not an observation. 
#There is ambigiouy in the FALSE outputs - more accurately, these are '?' states. I need to get the data for the observed absences.

# 1) Obtain absence data

### NEED NEW ABSENCE DATA

# 2) Down/Up sample accordingly. Downsample the training data and rename the default 'Class' to 'gcn_present'
table(train$gcn_present)
set.seed(100)
down_train <- downSample(x = train[, colnames(train) %ni% "gcn_present"],y = train$gcn_present)
colnames(down_train)[colnames(down_train)=="Class"] <- "gcn_present"
table(down_train$gcn_present)

# 3) Run analysis

logit_model <- glm(gcn_present~ pond_density + area_score + SI9_lee + SI9_oldham, family = "binomial", data=train)

summary(logit_model)

# 4) Predit, type must be 'response' for logit, taking the probabilty boundary to be 0.5

logit_predict <- predict(logit_model, newdata = test, type = "response")

y_pred_num <- ifelse(logit_predict > 0.5, 1, 0)
y_pred <- factor(y_pred_num, levels=c(0, 1))

y_act <- test$gcn_present

mean(y_pred == y_act)  # 70+%

score <- rowMeans(cbind(data$density_score, data$area_score,data$SI9_oldham), na.rm=TRUE)

score <- (data$density_score + data$area_score +  data$SI9_oldham)/3
plot(data$gcn_present ~ data$area_score, col="red4")

lines(gcn_present~ pond_density + area_score + SI9_lee, logit_predict, col="green4", lwd=2)

############### Load unknown GCN_PRESENT data

# 
# unknown_data <- read.csv('/Users/Mus/Downloads/final_table_all_info_unknown.csv',header=T,na.strings=c(""))
# 
# '%ni%' <- Negate('%in%')  # define 'not in' func
# options(scipen=999)  # prevents printing scientific notations.
# 
# logit_predict <- predict(logit_model, newdata = unknown_data, type = "response")
# y_pred_num <- ifelse(logit_predict > 0.5, 1, 0)
# y_pred <- factor(y_pred_num, levels=c(0, 1))
# 
# write.csv(y_pred, file = "/Users/Mus/Dropbox/Documents/Uni/Masters Docs/Absence Data/y_pred.csv")
