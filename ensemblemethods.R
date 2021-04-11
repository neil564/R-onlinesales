setwd('/Users/neilpatel/Desktop/Drexel Masters Files/R - College/Stat-642/Final Project')
getwd()
df <- read.csv(file = "OnlineSales.csv",  na.strings = c("", " "),stringsAsFactors = FALSE)

## Libraries
library(caret)
library(gmodels)
library(caretEnsemble)
library(Rcpp)
library(RcppZiggurat)
library(ggplot2)
library(corrplot)

#------------------------------------------
## Load libraries
library(rpart) # decision trees
library(rpart.plot) # decision tree plots


##Pre-Process
str(df)
summary(df)
any(is.na(df)) #no missing values
#df[duplicated(df), ] we have dupliacated values 
df <- df[!duplicated(df), ]
df$Revenue <- gsub(FALSE, 0, df$Revenue)
df$Revenue <- gsub(TRUE, 1, df$Revenue)
df$Weekend <- gsub(TRUE, 1, df$Weekend)
df$Weekend <- gsub(FALSE, 0, df$Weekend)
df$VisitorType <- gsub('Returning_Visitor', 0, df$VisitorType)
df$VisitorType <- gsub('New_Visitor', 1, df$VisitorType)
df$VisitorType <- gsub('Other', 2, df$VisitorType)

######### Target (Y) Variable ##################
df$Revenue <- factor(df$Revenue)

######### Categorical Variables  ##############
#Nominal (Unordered) Variables
noms <- c("OperatingSystems", "Browser", "Region", "TrafficType",
          'VisitorType','Weekend')
df[ ,noms] <- lapply(X = df[ ,noms], 
                     FUN = factor)

# Ordinal (Ordered) Variables
ords <- c("Month", "SpecialDay")
df[ ,ords] <- lapply(X = df[ ,ords], 
                     FUN = factor, 
                     ordered = TRUE)
df$Month <- factor(x = df$Month, 
                   levels = c("Jan","Feb", "Mar", "Apr","May", "June",
                              "Jul", "Aug", "Sep", "Oct", "Nov",  "Dec"),
                   ordered = TRUE)
unique(df[ ,"SpecialDay"]) #make sure its ordered correctly
df$Month  <- as.numeric(df$Month) #convert the months into numbers ordered
unique(df[ ,"Month"]) #make sure it worked

######## Numeric Variables #################
nums <- c('Administrative','Administrative_Duration','Informational',	'Informational_Duration',
          'ProductRelated',	'ProductRelated_Duration',	'BounceRates',	'ExitRates',	
          'PageValues')

vars <- c(noms, ords, nums)  #combine vectors

summary(df[,c(vars, "Revenue")])
plot(df$Revenue,
     main = "Revenue")
#imbalance so we must balance this 

## Training and Testing
set.seed(1112) #birthday seed Nov 12

# Create list of training indices
sub <- createDataPartition(y = df$Revenue, 
                           p = 0.80, 
                           list = FALSE)

train <- df[sub, ] # create training data
test <- df[-sub, ] # create testing data
target_var <- train$Revenue # identify target variable
weights <- c(sum(table(target_var))/(nlevels(target_var)*table(target_var)))
wghts <- weights[match(x = target_var, 
                       table = names(weights))]

ctrl <- trainControl(method = "cv", # k fold cross-validation
                     number = 5, # k = 5 folds
                     savePredictions = "final") # save final resampling summary measures only

set.seed(1112)

ensemble_models <- caretList(x = train[ ,vars], # predictor variables
                             y = train$Revenue, # target variable
                             trControl = ctrl,# control object
                             weights = wghts,
                             tuneList = list(bagging = caretModelSpec(method = "treebag"),
                                             randforest = caretModelSpec(method = "rf",
                                                                         tuneLength = 5)))
ensemble_models
plot(ensemble_models$randforest)
#running boosting takes to long, don't use 

results <- resamples(ensemble_models)
# Random Forest
ensemble_models[["randforest"]]$results
# Boosting
ensemble_models[["bagging"]]$results

# To view summary information for the best,
# tuned models for Accuracy and Kappa, we can 
# use the summary() function on our resamples 
# object.
summary(results)
plot(varImp(ensemble_models$bagging))
# Random Forest
plot(varImp(ensemble_models$randforest))
bwplot(results) 
dotplot(results)

##Testing Performance
ens_preds <- data.frame(predict(object = ensemble_models,
                                newdata = test),
                        stringsAsFactors = TRUE)
ph <- list()
for (i in 1:ncol(ens_preds)){
  ph[[i]] <- confusionMatrix(data = ens_preds[,i],
                             reference = test$Revenue,
                             positive = "1",
                             mode = "everything")
  names(ph)[[i]] <- colnames(ens_preds)[i]
}
## Overall
cbind(bag = ph[["bagging"]]$overall,
      rf = ph[["randforest"]]$overall)
## By Class
cbind(bag = ph[["bagging"]]$byClass,
      rf = ph[["randforest"]]$byClass)


## Goodness of Fit Training
t(matrix(colMeans(results$values[ ,-1]), 
         nrow = 2, 
         byrow = TRUE,
         dimnames = list(c("bag", "rf"), 
                         c("Accuracy", "Kappa"))))
## Goodness of Fit Testing
cbind(bag = ph[["bagging"]]$overall,
      rf = ph[["randforest"]]$overall)


save.image(file = "ensemblemethods.RData")


