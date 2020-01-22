######## Logit 1

#pond_density	density_score	pond_area	area_score	SI9_oldham	SI9_lee	presence

# 1, 3, 5 or 6.

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
trainDataIndex <- createDataPartition(data$gcn_present, p=0.85, list = F)  # 85% training data
train <- data[trainDataIndex, ]
test <- data[-trainDataIndex, ]

table(train$gcn_present)
set.seed(100)
down_train <- downSample(x = train[, colnames(train) %ni% "gcn_present"],y = train$gcn_present)
colnames(down_train)[colnames(down_train)=="Class"] <- "gcn_present"
table(down_train$gcn_present)

model <- glm(gcn_present~ pond_density + area_score + SI9_lee + SI9_oldham, data = train,family = binomial(link="logit"))
summary(model)

predicted.data <- data.frame(
  prob.gcn.present=model$fitted.values,
  gcn_present=train$gcn_present)

ggplot(data=predicted.data, aes(x=gcn_present, y=prob.gcn.present)) +
  geom_point(aes(color=gcn_present), size=5) +
  xlab("gcn_present") +
  ylab("Predicted probability of GCN Presence")

xtabs(~ prob.gcn.present + gcn_present, data=predicted.data)

predicted.data <- data.frame(
  prob.gcn.present=model$fitted.values,
  hd=train$gcn_present)

predicted.data <- predicted.data[
  order(predicted.data$prob.gcn.present, decreasing=FALSE),]
predicted.data$rank <- 1:nrow(predicted.data)

ggplot(data=predicted.data, aes(x=rank, y=prob.gcn.present)) +
  geom_point(aes(color=hd), alpha=1, shape=4, stroke=2) +
  xlab("Index") +
  ylab("prob.gcn.presence")

anova(model,test="Chisq")

pR2(model)

pred <- predict(model,newdata=test,type = "response")
summary(pred)
probs <- exp(pred)/(1+exp(pred))
y_pred_num <- ifelse(pred > 0.5, 1, 0)
y_pred <- factor(y_pred_num, levels=c(0, 1))
y_act <- test$gcn_present

mean(y_pred == y_act)  # 70+%


logodds <- predict(model, test, se.fit=TRUE, type = "response") #type="terms")
summary(logodds)


