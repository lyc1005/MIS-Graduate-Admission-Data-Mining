library(rvest)
library(tidyverse)
library(ggplot2)
library(lattice)
library(caret)
library(caretEnsemble)
library(Rcpp)
library(Amelia)
library(DMwR)
library(randomForest)
library(stringr)
library(maps)

setwd('C:\\Users\\Liu\\Desktop\\MIS graduate admission')
read.csv('Yocket-dataset.csv',header = TRUE)->dt
dim(dt)
names(dt)
table(dt$Status)

#preprocessing

dt[which(dt$Status %in% c('Admit','Reject')),]->dt_labeled
write.csv(dt_labeled,file = 'admission.csv')

read.csv('admission2.csv',header = TRUE)->dt2
#dt2$Undergrad_score
sub("CGPA","",dt2$Undergrad_score)->dt2$Undergrad_score
sub("%","",dt2$Undergrad_score)->dt2$Undergrad_score
as.numeric(dt2$Undergrad_score)->dt2$Undergrad_score
for(i in 1:nrow(dt2)) {if(dt2$Undergrad_score[i]>10){dt2$Undergrad_score[i]/10->dt2$Undergrad_score[i]}}

sub("[a-z]+","",dt2$work_ex) %>% as.numeric() -> dt2$work_ex
Season<-sub("[0-9]+","",dt2$Year) %>% str_trim()
dt2$Year<-sub("[a-zA-Z]+","",dt2$Year) %>% str_trim()
cbind.data.frame(dt2,Season)->dt2

dt2$work_ex[which(is.na(dt2$work_ex))]<-0
dt2$work_ex[which(dt2$work_ex==-1)]<-1

dt2$Test_score[which(is.na(dt2$Test_score))]<-112
dt2$Test_score[which(dt2$Eng_test=="IELTS")] %>% table()
dt2$Test_score[which(dt2$Test_score==0)]<-NA
dt2$Test_score[which(dt2$Test_score==6)]<-69
dt2$Test_score[which(dt2$Test_score==6.5)]<-86
dt2$Test_score[which(dt2$Test_score==7)]<-98
dt2$Test_score[which(dt2$Test_score==7.5)]<-106
dt2$Test_score[which(dt2$Test_score==8)]<-112

dt2[which(dt2$Year==2020),4]<-2019
2019-as.numeric(dt2$Year)->dt2$Year

summary(dt2)
names(dt2)

dt3<-dt2[,c(-1,-3,-13,-14,-15,-16)]     

missmap(dt3)
apply(is.na(dt3),2,sum)
apply(is.na(dt3),2,sum)/nrow(dt3)
summary(dt3)

#as.factor(dt3$Year)->dt3$Year
as.factor(dt3$Status)->dt3$Status
as.factor(dt3$Eng_test)->dt3$Eng_test
as.factor(dt3$Season)->dt3$Season
knnImputation(dt3)->dt4
missmap(dt4)
apply(is.na(dt4),2,sum)

#EDA
#Geography
qplot(Longitude,Latitude,data=dt2)+ borders("state",size=0.5)

data.frame(table(dt2$State))->mapstate

states <- map_data("state")
names(mapstate)<-c('region','freq')
mapstate$region <- tolower(mapstate$region)

choro <- merge(states,mapstate,by="region")
choro <- choro[order(choro$order),]
qplot(long,lat,data=choro,group=group,fill=-freq,geom="polygon")+borders("state",size=0.5)

#scores distribution
par(mfrow=c(1,3),mar=c(1.5,1.5,1,0))

hist(dt2$GRE_SCORE,main="GRE",col='red',freq = F)
lines(density(dt2$GRE_SCORE %>% na.omit()) ,col='blue',lwd=2)
hist(dt2[-which(dt2$Eng_test=='ENG TEST'),]$Test_score,main="TOEFL",col='red',freq = F)
lines(density(dt2[-which(dt2$Eng_test=='ENG TEST'),]$Test_score %>% na.omit()) ,col='blue',lwd=2)
hist(dt2$Undergrad_score,main="GRE",col='red',freq = F)
lines(density(dt2$Undergrad_score%>% na.omit()) ,col='blue',lwd=2)

read.csv('shujutoushi2.csv',header=TRUE)->ddt
parallelplot(ddt[,c(3,5,7,11)], groups=ddt$university)

#feature selection
summary(dt4)
RF<-randomForest(Status~.,data=dt4)
imp<-importance(RF)
varImpPlot(RF)
#PCA
princomp(dt4[,c(-1,-3,-5,-10,-11)],cor = TRUE)->PCA
names(PCA)
summary(PCA)
PCA$loadings
PCA$scores
screeplot(PCA)
biplot(PCA)


#let's build the midels！
#zero model
table(dt4$University) %>% data.frame()->university
university[which(university$Freq<15),]->university2
university2$Var1
dt4[-which(dt4$University %in% c('Boston University','Northwestern University','Pennsylvania State University',
          'Rutgers University-New Brunswick','University of California-Irvine',
          'University of California-Los Angeles','University of Delaware','University of Minnesota-Twin Cities',
          'University of Pennsylvania')),]->dt5
write.csv(dt5,'dt5.csv')
read.csv('dt5.csv',header = TRUE)->dt5         
dt5<-dt5[,-1]      
table(dt5$Status)/length(dt5$Status) #53.9%

#
set.seed(666)
createDataPartition(y=dt5$Status,p=0.75,list = FALSE)->inTrain
trainset<-dt5[inTrain,]
testset<-dt5[-inTrain,]

ctrl= trainControl(method = "repeatedcv",number = 5,repeats=3,search="random",
                   summaryFunction = twoClassSummary,
                   classProbs = TRUE, savePredictions = "final",allowParallel = TRUE)
model_list=caretList(
  Status~.,data=trainset,
  trControl=ctrl,
  metric="ROC",
  preProcess=c("center","scale"),
  methodList=c("glm","glmnet","pls","lda","svmLinear2",
               "svmRadial","knn","nb","rpart","rf","adaboost","xgbLinear","xgbTree"),
  tuneList = list(nnet=caretModelSpec(method="nnet",trace=F))
)                   
results <- resamples(model_list)
summary(results)
dotplot(results)
modelCor(results)
splom(results)

model_list2=caretList(
  Status~.,data=trainset,
  trControl=ctrl,
  metric="ROC",
  preProcess=c("center","scale"),
  methodList=c("rf","xgbLinear","svmLinear2","adaboost")
)
glm_ensemble <- caretStack(
  model_list2,
  method="glm",
  metric="ROC",
  trControl=ctrl
)
summary(glm_ensemble)

greedy_ensemble <- caretEnsemble(
  model_list2, 
  metric="ROC",
  trControl=ctrl)
summary(greedy_ensemble)

#validation
predict(model_list2[["rf"]],newdata=testset) -> pre.rf
predict(model_list2[["xgbLinear"]],newdata=testset) -> pre.xgb
predict(model_list2[["svmLinear2"]],newdata=testset) -> pre.svm
predict(model_list2[["adaboost"]],newdata=testset) -> pre.ada
predict(greedy_ensemble,newdata=testset) -> pre.greedy
predict(glm_ensemble,newdata=testset) -> pre.stacking

confusionMatrix(pre.rf,testset$Status)
confusionMatrix(pre.xgb,testset$Status)
confusionMatrix(pre.svm,testset$Status)
confusionMatrix(pre.ada,testset$Status)
confusionMatrix(pre.greedy,testset$Status) 
confusionMatrix(pre.stacking,testset$Status)
#hyperparameter tunning
rfGrid <- expand.grid(mtry = c(2:30))
nrow(rfGrid)
set.seed(825)
ctrl2= trainControl(method = "cv",number = 10,search="random",
                   summaryFunction = twoClassSummary,
                   classProbs = TRUE, savePredictions = "final",allowParallel = TRUE)
rfFit <- train(Status ~ ., data = trainset, 
                 method = "rf", 
                 trControl = ctrl2,
                 preProcess=c("center","scale"),
                 metric="ROC",tuneLength=30,
                 verbose = FALSE  )
 #tuneGrid = rfGrid
trellis.par.set(caretTheme())
plot(rfFit)  

svmFit <- train(Status ~ ., data = trainset, 
               method = "svmLinear2", 
               trace=FALSE,
               trControl = ctrl2,
               preProcess=c("center","scale"),
               metric="ROC",
               verbose = FALSE, tuneLength=30)
#tuneGrid = nnetGrid
trellis.par.set(caretTheme())
plot(svmFit)  

adaFit <- train(Status ~ ., data = trainset, 
                method = "adaboost", 
                trace=FALSE,
                trControl = ctrl2,
                preProcess=c("center","scale"),
                metric="ROC",
                verbose = FALSE, tuneLength=30)
#tuneGrid = nnetGrid
trellis.par.set(caretTheme())
plot(adaFit)  

model_list3=caretList(
  Status~.,data=trainset,
  trControl=ctrl2,
  trace=FALSE,
  metric="ROC",tuneLength=15,
  preProcess=c("center","scale"),
  methodList=c("rf","xgbLinear","svmLinear2"))
plot(model_list3$xgbLinear)  
glm_ensemble2 <- caretStack(
  model_list3,
  method="glm",
  metric="ROC",
  trControl=ctrl
)
greedy_ensemble2 <- caretEnsemble(
  model_list3, 
  metric="ROC",
  trControl=ctrl)


predict(rfFit,testset)->pre.rf_t
predict(svmFit,testset)->pre.svm_t
predict(greedy_ensemble2,testset)->pre.greedy_t
predict(glm_ensemble2,testset)->pre.glm_ensemble_t

confusionMatrix(pre.rf_t,testset$Status)
confusionMatrix(pre.svm_t,testset$Status)
confusionMatrix(pre.greedy_t,testset$Status)
confusionMatrix(pre.glm_ensemble_t,testset$Status)

#mistakes 
cbind.data.frame(pre.stacking,testset)->mis
mis[which(mis$pre.stacking!=mis$Status),]->mis2
mis[which(mis$pre.stacking==mis$Status),]->mis3
mis_FP<-mis2[which(mis2$pre.stacking=="Admit"),]
mis_FN<-mis2[which(mis2$pre.stacking=="Reject"),]
mis_TP<-mis3[which(mis3$pre.stacking=="Admit"),]
mis_TN<-mis3[which(mis3$pre.stacking=="Reject"),]
par(mfrow=c(1,3),mar=c(2,1,1,1))
color<-c('red','yellow','blue','green')
boxplot(mis_TP$GRE_SCORE,mis_TN$GRE_SCORE,mis_FP$GRE_SCORE,mis_FN$GRE_SCORE,main='GRE',col=color)
axis(side = 1,at=1:4,labels=c("TP","TN","FP","FN"))
boxplot(mis_TP$Test_score,mis_TN$Test_score,mis_FP$Test_score,mis_FN$Test_score,main='TOEFL',col=color)
axis(side = 1,at=1:4,labels=c("TP","TN","FP","FN"))
boxplot(mis_TP$Undergrad_score,mis_TN$Undergrad_score,mis_FP$Undergrad_score,mis_FN$Undergrad_score,main='GPA',col=color)
axis(side = 1,at=1:4,labels=c("TP","TN","FP","FN"))
