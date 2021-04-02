#################FINDING USEFUL COEFFICIENTS

library(tidyverse) 
library(glmnet)
library(Metrics)

#SIMPLE GLM MODEL#
model = glm(pstr ~ ageyear+SEX+higheduc+think_will_hospitalized_cv+fam_wage_loss_cv+fam_exp1_cv+fam_actions_cv___5+fam_actions_cv___6, data = trainData)

predictionTest = predict(model, testData)


testSubmission1 <- data.frame(test_id = testData$test_id, pstr = predictionTest )

testSubmission1$pstr[is.na(testSubmission1$pstr)] = mean(testSubmission1$pstr, na.rm =T)

write.csv(testSubmission1,"C:\\Users\\burto\\Desktop\\DataScienceCompetition\\testSubmission.csv", row.names = FALSE)


####FROM KAGGLE NOTEBOOK
list.files(path = "../input")

train = read.csv("../input/stressdata2/train.csv")

test = read.csv("../input/stressdata2/test.csv")

#split data for testing purposes

trainSplitSort = sort(sample(nrow(train), nrow(train)*.5))

trainSplit<-train[trainSplitSort,]

testSplit<-train[-trainSplitSort,]


#effectiveness testing for variables
y = train$pstr
x = train[,3:83]
x$pstr=NULL
x$SEX=as.numeric(as.factor(x$SEX))

pstr_table = train %>% group_by(higheduc) %>% summarise(mnpstr = mean(pstr))

x_join = right_join(x, pstr_table, by = "higheduc")

x_join$higheduc= NULL

x_join[is.na(x_join)] = 0
x_join = as.matrix(x_join)

model_glmn = glmnet::glmnet(x_join, y, alpha = 1)

sum(coef(model_glmn, s = 0.15)!=0)

coef(model_glmn, s=.15)

#DATA SPLIT FOR TESTING PURPOSES

trainSplitSort = sort(sample(nrow(train), nrow(train)*.5))

trainSplit<-train[trainSplitSort,]

testSplit<-train[-trainSplitSort,]


############################
#######GBM MODEL############

library(gbm)

#testing portion for local cross validation
gbmModel = gbm(formula = pstr ~ hincome+child_social_media_time_cv+physical_activities_hr_cv, data = train)

gbmPrediction = predict(gbmModel, test, n.trees = 100)


gbmModelComp <- data.frame(test_id = test$test_id, pstr = gbmPrediction)

gbmModelComp$pstr[is.na(gbmModelComp$pstr)] = mean(gbmModelComp$pstr, na.rm = T)

#rmse(testSplit$pstr, gbmModelComp$pstr)

write.csv(gbmModelComp,"gbmModelCompSub.csv", row.names = FALSE)
############################
#RANDOM FOREST model with updated variables

install.packages("randomForest")
library(randomForest)

rfModel = randomForest(pstr ~ hincome+child_social_media_time_cv+physical_activities_hr_cv, data = trainSplit, ntree =100,mtry=2,importance = TRUE, na.action = na.exclude)

rfPrediction = predict(rfModel, trainSplit)


rfModelComp <-data.frame(test_id = trainSplit$train_id, pstr = rfPrediction)

rfModelComp$pstr[is.na(rfModelComp$pstr)] = mean(rfModelComp$pstr, na.rm = T)

library(Metrics)
#testing rmse for rfModel
rmse(testSplit$pstr, rfModelComp$pstr)

#submission for random forest model with updated variables (probaly worse than before)

rfSub = predict(rfModel, testData)

rfModelCompSub <-data.frame(test_id = testData$test_id, pstr = rfSub)

rfModelCompSub$pstr[is.na(rfModelCompSub$pstr)] = mean(rfModelCompSub$pstr, na.rm = T)


write.csv(rfModelCompSub,"C:\\Users\\burto\\Desktop\\DataScienceCompetition\\rfModelUpdatedVariables.csv", row.names = FALSE)

#############################################
###############MODEL STACKING################


#split data for stacking
index = sample(nrow(train), nrow(train)/2, F)

train1 = train[index,]
train2 = train[index,]

#base model

model_glm = glm(pstr ~ hincome+child_social_media_time_cv+physical_activities_hr_cv, data = train1)
train2$glm_prediction = predict(model_glm, train2)
test$glm_prediction = predict(model_glm, test)

#random forest model

library(randomForest)
model_rf = randomForest(pstr ~ hincome+child_social_media_time_cv+physical_activities_hr_cv, data = train1, ntree =100,mtry=2,importance = TRUE, na.action = na.exclude)
train2$rf_prediction = predict(model_rf, train2)
test$rf_prediction = predict(model_rf, test)

#gbm model
model_gbm = gbm(formula = pstr ~ hincome+child_social_media_time_cv+physical_activities_hr_cv, data = train1, n.trees = 100)
train2$gbm_prediction = predict(model_gbm, train2, n.trees = 100)
test$gbm_prediction = predict(model_gbm, test, n.trees = 100)



plot(train2$glm_prediction, train2$rf_prediction)

#stack time

model_stack = glm(pstr~ hincome*glm_prediction+child_social_media_time_cv*glm_prediction
                  +physical_activities_hr_cv*glm_prediction + hincome*gbm_prediction+child_social_media_time_cv*gbm_prediction
                  +physical_activities_hr_cv*gbm_prediction + hincome*rf_prediction+child_social_media_time_cv*rf_prediction
                  +physical_activities_hr_cv*rf_prediction, data = train2)

stack_predict = predict(model_stack, test)




stackComp <- data.frame(test_id = test$test_id, pstr = stack_predict)

stackComp$pstr[is.na(stackComp$pstr)] = mean(stackComp$pstr, na.rm = T)

write.csv(stackComp,"stackModel.csv", row.names = FALSE)



#####################################
###########KERAS ATTEMPT#############
#KERAS ATTEMPT
library(keras)
library(tensorflow)
library(imager)

# create the model
model <- keras_model_sequential()
model %>% 
  layer_dense(units = 100, activation = 'relu', input_shape = c(80)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 50, activation = 'relu') %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 1, activation = 'linear')

summary(model)


# compile model and specify optimization
model %>% compile(
  loss = 'MeanSquaredError',
  optimizer = optimizer_adam(),
  metrics = c('MeanSquaredError')
)


# fit model
history <- model %>% fit(
  x_join, y, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
)

plot(history)




