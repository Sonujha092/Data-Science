# Importing the dataset
df <- read.csv("indian_liver_patient.csv")
attach(df)
colnames(df)[colnames(df) == "Dataset"] <- "Disease"

# Converting Dataset from numeric to factor
df$Disease= as.factor(df$Disease)

# Treating missing value
colSums(sapply(df,is.na))
df$Albumin_and_Globulin_Ratio = ifelse(is.na(df$Albumin_and_Globulin_Ratio)==T,
                                       mean(df$Albumin_and_Globulin_Ratio,na.rm = T),
                                       df$Albumin_and_Globulin_Ratio)
# Outlier Treatement
boxplot(df)
boxplot(Total_Protiens)
qnt = quantile(Total_Protiens,seq(0,1,by = .05))
upper <- qnt[16]+1.5*IQR(df$bmi)
lower <- qnt[6]+1.5*IQR(df$bmi)
df$Total_Protiens = ifelse(df$Total_Protiens>qnt[16],upper,df$Total_Protiens)
df$Total_Protiens = ifelse(df$Total_Protiens<qnt[6],lower,df$Total_Protiens)
boxplot(df$Total_Protiens)

# Alamine_Aminotransferese
library(ggplot2)
ggplot(df, aes(x=Disease, y= Alkaline_Phosphotase)) + 
  geom_boxplot()

ggplot(df, aes(x=Disease, y= Alamine_Aminotransferase)) + 
  geom_boxplot()
boxplot(df)

# Splitting the dataset into training and test set
library(caTools)
split = sample.split(df$Disease, SplitRatio = 0.75)
training_set = subset(df, split == TRUE)
test_set = subset(df, split == FALSE)

# Fitting Logistic Regressor to training data
classifier = glm(formula = Disease~.,family = binomial,data = training_set)
summary(classifier)

# Predicting the Test set results
prob_pred = predict(classifier, type = 'response', newdata = test_set[-11])
y_pred = ifelse(prob_pred > 0.5, 1, 0)


# Making the Confusion Matrix
cm = table(test_set[, 11], y_pred > 0.5)
Acc(classifier)

# ROC curve
library(ROCR)
predictTrain = predict(classifier,test_set,type = "response")
ROCRpred = prediction(predictTrain,test_set$Disease)
ROCRpref = performance(ROCRpred,"tpr","fpr")
plot(ROCRpref)

# Assumption
df1 = df[-11]
df1 = df1[-2]
cor(df1)

# Model 2
classifier1 = glm(Disease ~ .-Direct_Bilirubin-Aspartate_Aminotransferase,family = "binomial",data = training_set)
summary(classifier1)

# Predicting the Test set results
prob_pred1 = predict(classifier1, type = 'response', newdata = test_set[-11])
y_pred1 = ifelse(prob_pred > 0.5, 1, 0)


# Making the Confusion Matrix
cm1 = table(test_set[, 11], y_pred1 > 0.5)
Acc(classifier1)

# Stepwise Model
classifier2 = step(glm(Disease~.,family = binomial,data = training_set),
                   direction = "backward")
summary(classifier2)
prob_pred2 = predict(classifier2, type = 'response', newdata = test_set[-11])
y_pred2 = ifelse(prob_pred > 0.5, 1, 0)

# Making the Confusion Matrix
cm2 = table(test_set[, 11], y_pred2 > 0.5)
Acc(classifier2)


svm_model1 <- svm(Disease ~ .,data = training_set)
summary(svm_model1)

# table(ypred,training$Outcome)
confusionMatrix(svm_model1$fitted,training_set$Disease)

# Accuracy on testing data set 
ypred1 = predict(svm_model1,test_set)

# table(ypred1,testing$outcome)
confusionMatrix(ypred1,test_set$Disease)
