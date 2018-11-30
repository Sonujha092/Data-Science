inputData <- read.csv("adult.csv")
head(inputData)
colnames(inputData)[colnames(inputData) == "income"] <- "above50k"
inputData$above50k = as.numeric(inputData$above50k)
inputData$above50k = ifelse(inputData$above50k==1,0,1)

table(inputData$above50k)

# Create Training Data
input_ones <- inputData[which(inputData$above50k == 1), ]
input_zeros <- inputData[which(inputData$above50k == 0), ]

input_ones_training_rows <- sample(1:nrow(input_ones), 0.7*nrow(input_ones))  
input_zeros_training_rows <- sample(1:nrow(input_zeros), 0.7*nrow(input_ones))
training_ones <- input_ones[input_ones_training_rows, ]  
training_zeros <- input_zeros[input_zeros_training_rows, ]
trainingData <- rbind(training_ones, training_zeros)

# Create Test Data
test_ones <- input_ones[-input_ones_training_rows, ]
test_zeros <- input_zeros[-input_zeros_training_rows, ]
testData <- rbind(test_ones, test_zeros)

# 
library(smbinning)
# segregating continuous and categorical variable
str(inputData)
factor_vars <- c('workclass','education','marital.status','occupation',
                 'relationship','race','sex','native.country')
continuous_vars <- c('age','fnlwgt','education.num','caputal.gain','capital.coss',
                     'hours.per.week')
iv_df <- data.frame(VARS = c(factor_vars,continuous_vars),IV = numeric(14))

# compute IV for categoricals
for(factor_var in factor_vars){
  smb <- smbinning.factor(trainingData, y="above50k", x=factor_var,maxcat = 16)  # WOE table
  if(class(smb) != "character"){ # heck if some error occured
    iv_df[iv_df$VARS == factor_var, "IV"] <- smb$iv
  }
}

# compute IV for continuous vars
for(continuous_var in continuous_vars){
  smb <- smbinning(trainingData, y="above50k", x=continuous_var)  # WOE table
  if(class(smb) != "character"){
    iv_df[iv_df$VARS == continuous_var, "IV"] <- smb$iv
  }
}

iv_df <- iv_df[order(-iv_df$IV), ]
iv_df

# Building model and predict
logitMod <- glm(above50k ~ relationship + age + occupation + education,
                data=trainingData, family=binomial)
predicted <- predict(logitMod, testData, type="response") 

library(InformationValue)
#install.packages('InformationValue')
optCutOff <- optimalCutoff(testData$above50k, predicted)[1]
summary(logitMod)

vif(logitMod)
misClassError(testData$above50k, predicted, threshold = optCutOff)
plotROC(testData$above50k, predicted)

Concordance(testData$above50k, predicted)

sensitivity(testData$above50k, predicted, threshold = optCutOff)
specificity(testData$above50k, predicted, threshold = optCutOff)

confusionMatrix(testData$above50k, predicted, threshold = optCutOff)

