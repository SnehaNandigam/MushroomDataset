library(caret)              #install.packages("caret") if package is not installed
library(ggplot2)            #install.packages("ggplot2") if package is not installed
library(randomForest)       #install.packages("randomForest") if package is not installed
library(rpart.plot)         #install.packages("rpart.plot") if package is not installed
library(e1071)              #install.packages("e1071") if package is not installed
mushrooms <- read.csv("C:/Users/Sneha/Downloads/mushrooms.csv")#Change path to the location where the CSV file is stored
#https://www.kaggle.com/uciml/mushroom-classification--link to the kaggle dataset
View(mushrooms)             #To view the data
summary(mushrooms)          #Summary statistics of the dataset
for(i in seq(2,23,1)){plot(mushrooms[,i],mushrooms[,1],xlab=colnames(mushrooms)[i],ylab=colnames(mushrooms)[1])}#all composition plots
for(i in seq(1,23,1)){plot(mushrooms[,i],xlab=colnames(mushrooms)[i],main="Composition")}#find the composition of dataset
set.seed(1,sample.kind="Rounding")
testIndex<-createDataPartition(mushrooms$class,times=1,p=0.3,list=FALSE)#To create a trainset and a testset
testSet2<-mushrooms[testIndex,]
trainSet2<-mushrooms[-testIndex,]
fit<-lm(class~cap.shape+cap.surface+cap.color+bruises+odor+gill.attachment+gill.spacing+gill.size+gill.color+stalk.shape+stalk.root+stalk.surface.above.ring+stalk.surface.below.ring+stalk.color.above.ring+stalk.color.below.ring+veil.color+ring.number+ring.type+spore.print.color+population+habitat,data=trainSet2)#Linear Regression Model
predicted<-predict(fit,testSet2)            #Predicting the testset values using the fit
predicted<-as.integer(as.numeric(as.character(predicted))*1000)/1000
predicted<-ifelse(predicted>1.5,2,1)        #Using probability to determine the class
mean(predicted==as.numeric(testSet2$class)) # Accuracy for Linear Regression Model
plot(fit$coefficients,main="Coefficient Range for Linear Regression")#Plot to visualize the coefficient range
importance<-as.data.frame(fit$coefficients)
View(importance[is.na(importance[,1]),0])   #Shows that no predictor can be removed from the model



#RANDOM FOREST
rfit<-randomForest(class~.,data=trainSet2)                 #Model for Random Forest
mean(predict(rfit,testSet2)==testSet2$class)               #Measuring accuracy of prediction
plot(rfit$importance,main="Importance for Random Forest")  #Plot to show importance of predictors in the model  

#DECISION TREE
dfit<-rpart(class~.,data=trainSet2,method="class")    #Model for Decision Tree
pred<-predict(dfit,testSet2)                          #Predicted Probability values 
pred<-ifelse(pred[,1]>0.5,'e','p')                    #Determine class using the probabilities
mean(pred==testSet2$class)                            #Accuracy of Decision Tree Model
print(dfit)                                           #summary of the tree
rpart.plot(dfit,main="Decision Tree")                 #Plot the Decision Tree
rpart.rules(dfit)                                     #To print the rules



#SVM
trainSet3<-trainSet2[,-17]            #Removing Predictor Variable with no Variance
svmfit<-svm(class~.,data=trainSet3)   #Model for SVM
svmpred<-predict(svmfit,testSet2)     #Predicting the class Values
mean(svmpred==testSet2$class)         #Accuracy of SVM Model


#NAIVE BAYES
nvfit<-naiveBayes(class~.,data=trainSet2)  #Model for Naive Bayes
nvpred<-predict(nvfit,testSet2)            #Predicting the class Values
mean(nvpred==testSet2$class)               #Accuracy of Naive Bayes Model

accuracies<-as.data.frame(t(data.frame('Linear Regression','Random Forest','Decision Tree','SVM','Naive Bayes')))
accuracies$acc[1]<-mean(predicted==as.numeric(testSet2$class))
accuracies$acc[2]<-mean(predict(rfit,testSet2)==testSet2$class)
accuracies$acc[3]<-mean(pred==testSet2$class)
accuracies$acc[4]<-mean(svmpred==testSet2$class) 
accuracies$acc[5]<-mean(nvpred==testSet2$class)
View(accuracies)                   #Data Frame to compare the accuracies of each of the model applied
