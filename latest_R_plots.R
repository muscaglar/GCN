library(rms)library("caret")library(pscl)library('glmnet')library("klaR")library("h2o")library("magrittr")
library("dplyr")library("pillar")library(mlr)library(ggplot2)library(popbio)library(InformationValue)library(ROCR)
library(jtools)

data <- read.csv('/Desktop/known_all_factors.csv',header=T,na.strings=c(""))
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

##################################################################### Models
model1 <- glm(gcn_present ~ pond_density, family = binomial(link = "logit"), data=down_train) #+ area_score + SI9_lee + SI9_oldham
model2 <- glm(gcn_present ~ area_score, family = binomial(link = "logit"), data=down_train) #+ area_score + SI9_lee + SI9_oldham
model3 <- glm(gcn_present ~ SI9_lee, family = binomial(link = "logit"), data=down_train) #+ area_score + SI9_lee + SI9_oldham
model4 <- glm(gcn_present ~ SI9_oldham, family = binomial(link = "logit"), data=down_train) #+ area_score + SI9_lee + SI9_oldham
model5 <- glm(gcn_present ~ pond_area, family = binomial(link = "logit"), data=down_train) #+ area_score + SI9_lee + SI9_oldham
#######################################################################################

export_summs(model1, model2, model3, model4, scale = TRUE,results = 'asis', error_format = "[{conf.low}, {conf.high}]", to.file = "PDF", file.name = "test.pdf")

##################################################################### Downtrained plots
effect_plot(model1, pred = pond_density, interval = TRUE)
effect_plot(model2, pred = area_score, interval = TRUE)
effect_plot(model3, pred = SI9_lee, interval = TRUE)
effect_plot(model4, pred = SI9_oldham, interval = TRUE)
effect_plot(model5, pred = pond_area, interval = TRUE)

plot(down_train$SI9_lee,as.numeric(down_train$gcn_present)-1)
plot(down_train$SI9_oldham,as.numeric(down_train$gcn_present)-1)
plot(down_train$area_score,as.numeric(down_train$gcn_present)-1)
plot(down_train$pond_density,as.numeric(down_train$gcn_present)-1)
plot(down_train$pond_area,as.numeric(down_train$gcn_present)-1)

ggplot(down_train, aes(pond_density, as.numeric(gcn_present)-1)) + stat_smooth(method="glm", method.args = list(family="binomial"), formula=y~x,alpha=0.2, size=2) + geom_point(position=position_jitter(height=0.03, width=0)) + xlab("Pond Density") + ylab("GCN Presence")
ggplot(down_train, aes(area_score, as.numeric(gcn_present)-1)) + stat_smooth(method="glm", method.args = list(family="binomial"), formula=y~x,alpha=0.2, size=2) + geom_point(position=position_jitter(height=0.03, width=0)) + xlab("Pond Area Score") + ylab("GCN Presence")
ggplot(down_train, aes(SI9_lee, as.numeric(gcn_present)-1)) + stat_smooth(method="glm", method.args = list(family="binomial"), formula=y~x,alpha=0.2, size=2) + geom_point(position=position_jitter(height=0.03, width=0)) + xlab("SI9 Lee") + ylab("GCN Presence")
ggplot(down_train, aes(SI9_oldham, as.numeric(gcn_present)-1)) + stat_smooth(method="glm", method.args = list(family="binomial"), formula=y~x,alpha=0.2, size=2) + geom_point(position=position_jitter(height=0.03, width=0)) + xlab("SI9 Oldham") + ylab("GCN Presence")
ggplot(down_train, aes(pond_area, as.numeric(gcn_present)-1)) + scale_x_log10()+ stat_smooth(method="glm", method.args = list(family="binomial"), formula=y~x,alpha=0.2, size=2) + geom_point(position=position_jitter(height=0.03, width=0)) + xlab("Pond Area") + ylab("GCN Presence")
#######################################################################################

print(summary(model))
summ(model)

##################################################################### Prediction Models
p1 <- predict(model1, newdata=test, type="response")
p2 <- predict(model2, newdata=test, type="response")
p3 <- predict(model3, newdata=test, type="response")
p4 <- predict(model4, newdata=test, type="response")
p5 <- predict(model5, newdata=test, type="response")
#######################################################################################

##################################################################### MisClassError
fitted.results <- ifelse(p1 > 0.5,1,0)
misClasificError <- mean(fitted.results != test$gcn_present)
print(paste('Accuracy',1-misClasificError))

fitted.results <- ifelse(p2 > 0.5,1,0)
misClasificError <- mean(fitted.results != test$gcn_present)
print(paste('Accuracy',1-misClasificError))

fitted.results <- ifelse(p3 > 0.5,1,0)
misClasificError <- mean(fitted.results != test$gcn_present)
print(paste('Accuracy',1-misClasificError))

fitted.results <- ifelse(p4 > 0.5,1,0)
misClasificError <- mean(fitted.results != test$gcn_present)
print(paste('Accuracy',1-misClasificError))

fitted.results <- ifelse(p5 > 0.5,1,0)
misClasificError <- mean(fitted.results != test$gcn_present)
print(paste('Accuracy',1-misClasificError))
#######################################################################################

pr1 <- prediction(p1, test$gcn_present)
prf1 <- performance(pr1, measure = "tpr", x.measure = "fpr")
plot(prf1)
auc <- performance(pr1, measure = "auc")
auc <- auc@y.values[[1]]
auc
optCutOff <- optimalCutoff(test$gcn_present, p1)[1] 
optCutOff
vif(model1)
misClassError(test$gcn_present, p1, threshold = optCutOff)
plotROC(test$gcn_present, p1)
Concordance(test$gcn_present, p1)
confusionMatrix(test$gcn_present, p1, threshold = optCutOff)
sensitivity(test$gcn_present, p1, threshold = optCutOff)
specificity(test$gcn_present, p1, threshold = optCutOff)

pr2 <- prediction(p2, test$gcn_present)
prf2 <- performance(pr2, measure = "tpr", x.measure = "fpr")
plot(prf2)
auc <- performance(pr2, measure = "auc")
auc <- auc@y.values[[1]]
auc
optCutOff <- optimalCutoff(test$gcn_present, p2)[1] 
optCutOff
vif(model2)
misClassError(test$gcn_present, p2, threshold = optCutOff)
plotROC(test$gcn_present, p2)
Concordance(test$gcn_present, p2)
confusionMatrix(test$gcn_present, p2, threshold = optCutOff)
sensitivity(test$gcn_present, p2, threshold = optCutOff)
specificity(test$gcn_present, p2, threshold = optCutOff)

pr3 <- prediction(p3, test$gcn_present)
prf3 <- performance(pr3, measure = "tpr", x.measure = "fpr")
plot(prf3)
auc <- performance(pr3, measure = "auc")
auc <- auc@y.values[[1]]
auc
optCutOff <- optimalCutoff(test$gcn_present, p3)[1] 
optCutOff
vif(model3)
misClassError(test$gcn_present, p3, threshold = optCutOff)
plotROC(test$gcn_present, p3)
Concordance(test$gcn_present, p3)
confusionMatrix(test$gcn_present, p3, threshold = optCutOff)
sensitivity(test$gcn_present, p3, threshold = optCutOff)
specificity(test$gcn_present, p3, threshold = optCutOff)

pr4 <- prediction(p4, test$gcn_present)
prf4 <- performance(pr4, measure = "tpr", x.measure = "fpr")
plot(prf4)
auc <- performance(pr4, measure = "auc")
auc <- auc@y.values[[1]]
auc
optCutOff <- optimalCutoff(test$gcn_present, p4)[1]
optCutOff
vif(model4)
misClassError(test$gcn_present, p4, threshold = optCutOff)
plotROC(test$gcn_present, p4)
Concordance(test$gcn_present, p4)
confusionMatrix(test$gcn_present, p4, threshold = optCutOff)
sensitivity(test$gcn_present, p4, threshold = optCutOff)
specificity(test$gcn_present, p4, threshold = optCutOff)

explanatory = c("pond_density", "area_score","SI9_lee", "SI9_oldham")
table1 <- finalfit(down_train,'gcn_present', explanatory, metrics=TRUE)
save(table1, explanatory, file = "out.rda")

