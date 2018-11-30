# Bank Marketing
# Importing the dataset
df <- read.csv('bank.csv',sep = ";")
attach(df)
dim(df)
names(df)
str(df)
summary(df)
# Categorical variables : job,marital,default,housing,loan,contact,poutcome,month,day
# Ordinal variable: education
# Continuous variable : age,balance,duration,previous


# univariate analysis
table(y)
df$y = as.numeric(df$y)
df$y = ifelse(df$y == 1,0,1)

#Bad rate
(prop.table(table(y))*100)

# Bivariate analysis

table(df$job,df$y)
table(df$marital,df$y)
table(df$education,df$y)
table(df$default,df$y)
table(df$housing,df$y)
table(df$loan,df$y)
table(df$contact,df$y)
table(df$campaign,df$y)
table(df$poutcome,df$y)

# Creating dummy variable for categorical variable
#install.packages('fastDummies')
results <- fastDummies::dummy_cols(df,remove_first_dummy = TRUE)

# Splitting the dataset into training and test set
library(caTools)
split = sample.split(df$y,SplitRatio = .7)
training_set = subset(df,split==TRUE)
test_set = subset(df,split==FALSE)

# Fitting logistic regression to the dataset
classifier = step(glm(y~.,data=training_set),direction = 'both')
summary(classifier)

# predictiong on test set
prob_pred = predict(classifier,test_set)
y_pred = ifelse(prob_pred>.5,1,0)

# Confusion Matrix
table(test_set$y,y_pred)

#ROC curve
library(ROCR)
predictTrain = predict(classifier,test_set,type = "response")
ROCRpred = prediction(predictTrain,test_set$y)
ROCRpref = performance(ROCRpred,"tpr","fpr")
plot(ROCRpref)
pred = prediction(prob_pred,test_set$y)
as.numeric(performance(pred,"auc")@y.values)
