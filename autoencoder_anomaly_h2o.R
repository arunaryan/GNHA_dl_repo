# install.packages("h2o")
options(java.parameters = "-Xmx8192m")
library(readr)
library(pROC)
library(tidyr)
library(Matrix)
library(caret)
library(reshape)
library(lubridate)
library(dplyr)
library(xgboost)
library(h2o)
# point to the prostate data set in the h2o folder - no need to load h2o in memory yet
prosPath = system.file("extdata", "prostate.csv", package = "h2o")
prostate_df <- read.csv(prosPath)

# We don't need the ID field
prostate_df <- prostate_df[,-1]
summary(prostate_df)

set.seed(1234)
random_splits <- runif(nrow(prostate_df))
train_df <- prostate_df[random_splits < .5,]
dim(train_df)

validate_df <- prostate_df[random_splits >=.5,]
dim(validate_df)

# Get benchmark score

# install.packages('randomForest')
library(randomForest)

outcome_name <- 'CAPSULE'
feature_names <- setdiff(names(prostate_df), outcome_name)
set.seed(1234)
rf_model <- randomForest(x=train_df[,feature_names],
                         y=as.factor(train_df[,outcome_name]),
                         importance=TRUE, ntree=20, mtry = 3)

validate_predictions <- predict(rf_model, newdata=validate_df[,feature_names], type="prob")


# install.packages('pROC')
library(pROC)
auc_rf = roc(response=as.numeric(as.factor(validate_df[,outcome_name]))-1,
             predictor=validate_predictions[,2])

plot(auc_rf, print.thres = "best", main=paste('AUC:',round(auc_rf$auc[[1]],3)))
abline(h=1,col='blue')
abline(h=0,col='green')

# build autoencoder model

library(h2o)
localH2O = h2o.init()
prostate.hex<-as.h2o(prostate_df, destination_frame="train.hex")
prostate.dl = h2o.deeplearning(x = feature_names, training_frame = prostate.hex,
                               autoencoder = TRUE,
                               reproducible = T,
                               seed = 1234,
                               hidden = c(6,5,6), epochs = 50)

# interesting per feature error scores
 prostate.anon = h2o.anomaly(prostate.dl, prostate.hex, per_feature=TRUE)
 head(prostate.anon)

prostate.anon = h2o.anomaly(prostate.dl, prostate.hex, per_feature=FALSE)
head(prostate.anon)
err <- as.data.frame(prostate.anon)

# interesting reduced features (defaults to last hidden layer)
# http://www.rdocumentation.org/packages/h2o/functions/h2o.deepfeatures
reduced_new  <- h2o.deepfeatures(prostate.dl, prostate.hex,layer=2)

plot(sort(err$Reconstruction.MSE))

# use the easy portion and model with random forest using same settings

train_df_auto <- train_df[err$Reconstruction.MSE <= 0.1,]
train_df_auto<-train_df_auto[complete.cases(train_df_auto),]

set.seed(1234)
rf_model <- randomForest(x=train_df_auto[,feature_names],
                         y=as.factor(train_df_auto[,outcome_name]),
                         importance=TRUE, ntree=20, mtry = 3)

validate_predictions_known <- predict(rf_model, newdata=validate_df[,feature_names], type="prob")

auc_rf = roc(response=as.numeric(as.factor(validate_df[,outcome_name]))-1,
             predictor=validate_predictions_known[,2])

plot(auc_rf, print.thres = "best", main=paste('AUC:',round(auc_rf$auc[[1]],3)))
abline(h=1,col='blue')
abline(h=0,col='green')

# use the hard portion and model with random forest using same settings
train_df_auto <- train_df[err$Reconstruction.MSE >= 0.1,]
train_df_auto<-train_df_auto[complete.cases(train_df_auto),]
set.seed(1234)
rf_model <- randomForest(x=train_df_auto[,feature_names],
                         y=as.factor(train_df_auto[,outcome_name]),
                         importance=TRUE, ntree=20, mtry = 3)

validate_predictions_unknown <- predict(rf_model, newdata=validate_df[,feature_names], type="prob")
auc_rf = roc(response=as.numeric(as.factor(validate_df[,outcome_name]))-1,
             predictor=validate_predictions_unknown[,2])

plot(auc_rf, print.thres = "best", main=paste('AUC:',round(auc_rf$auc[[1]],3)))
abline(h=1,col='blue')
abline(h=0,col='green')

# bag both results set and measure final AUC score
valid_all <- (validate_predictions_known[,2] + validate_predictions_unknown[,2]) / 2
auc_rf = roc(response=as.numeric(as.factor(validate_df[,outcome_name]))-1,
             predictor=valid_all)

plot(auc_rf, print.thres = "best", main=paste('AUC:',round(auc_rf$auc[[1]],3)))
abline(h=1,col='blue')
abline(h=0,col='green')

require(ggplot2)
prostate_df$err<-err$Reconstruction.MSE
prostate_df$class<-ifelse(prostate_df$err>=0.1,1,0)

prostate_df$RACE<-as.factor(prostate_df$RACE)
prostate_df$CAPSULE<-as.factor(prostate_df$CAPSULE)
prostate_df$class<-as.factor(prostate_df$class)
ggplot(prostate_df, aes(AGE,fill=class)) + geom_histogram(alpha = 0.5, aes(y = ..density..), position = 'identity')
ggplot(prostate_df, aes(AGE, fill = CAPSULE)) + geom_histogram(alpha = 0.5, aes(y = ..density..), position = 'identity')
ggplot(prostate_df, aes(AGE, fill = class)) + geom_density(alpha = 0.2)
ggplot(prostate_df, aes(AGE, fill = CAPSULE)) + geom_density(alpha = 0.2)
ggplot(prostate_df, aes(PSA, fill = class)) + geom_density(alpha = 0.2)
ggplot(prostate_df, aes(PSA, fill = CAPSULE)) + geom_density(alpha = 0.2)
ggplot(prostate_df, aes(VOL, fill = class)) + geom_density(alpha = 0.2)
ggplot(prostate_df, aes(VOL, fill = CAPSULE)) + geom_density(alpha = 0.2)
ggplot(prostate_df, aes(DPROS, fill = class)) + geom_density(alpha = 0.2)
ggplot(prostate_df, aes(DPROS, fill = CAPSULE)) + geom_density(alpha = 0.2)
ggplot(prostate_df, aes(DCAPS, fill = class)) + geom_density(alpha = 0.2)
ggplot(prostate_df, aes(DCAPS, fill = CAPSULE)) + geom_density(alpha = 0.2)
ggplot(prostate_df, aes(GLEASON, fill = class)) + geom_density(alpha = 0.2)
ggplot(prostate_df, aes(GLEASON, fill = CAPSULE)) + geom_density(alpha = 0.2)
ggplot(prostate_df, aes(err, fill = class)) + geom_density(alpha = 0.2)
ggplot(prostate_df, aes(err, fill = CAPSULE)) + geom_density(alpha = 0.2)
plot(density(log(err$Reconstruction.MSE+1)))
#, fill = RACE

# library(wordcloud2)
# exc_hcc<-data.frame(hcc=excellus_cst_2018_1_admt_vs_diag_v1_0$hccdesc,freq = excellus_cst_2018_1_admt_vs_diag_v1_0$AdmitCnt,stringsAsFactors = F)
# exc_diags<-data.frame(hcc=excellus_cst_2018_1_admt_vs_diag_v1_1$diagdesc,freq = excellus_cst_2018_1_admt_vs_diag_v1_1$AdmitCnt,stringsAsFactors = F)
# 
# wordcloud2(exc_hcc[exc_hcc$hcc!="NULL",], size=0.25, color='random-dark')
# wordcloud2(exc_diags, size=0.25, color='random-dark')


library(mixtools)
library(stats)
library(plotly)


#cohort2014<-train[train$PMPM>0,c("DWMemberID","PMPM","Y2PMPM","AdmitCnt","Y2AdmitCnt","ADK","Y2ADK")]
x=as.matrix((log(err$Reconstruction.MSE)))

err_npEM<-normalmixEM(x) #, bw = bw.nrd0(as.vector(as.matrix(x))), samebw = TRUE, eps = 1e-6, 
                      #maxiter = 1000, stochastic = FALSE, verb = TRUE) 
plot(err_npEM,which=2)
lines(density((log(err$Reconstruction.MSE))), lty=2, lwd=2)
#lines(density(prostate_df$VOL), lty=2, lwd=2)
err$cbot.score<-1-err_npEM$posterior[,1]

tail(sort(rf_model$importance), 20)

