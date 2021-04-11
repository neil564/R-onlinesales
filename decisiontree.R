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


##TEST 1
FD.rpart <- rpart(formula = Revenue ~ ., 
                  data = train[ ,c(vars, "Revenue")],
                  method = 'class')

FD.rpart
FD.rpart$variable.importance

prp(x = FD.rpart,
    extra = 2) 

base.trpreds <- predict(object = FD.rpart,
                        newdata = train, 
                        type = "class")

DT_train_conf <- confusionMatrix(data = base.trpreds, # predictions
                                 reference = train$Revenue, # actual
                                 positive = "1",
                                 mode = "everything")
DT_train_conf
## Training Performance
base.trpreds <- predict(object = FD.rpart,
                        newdata = train, 
                        type = "class")
DT_train_conf <- confusionMatrix(data = base.trpreds, # predictions
                                 reference = train$Revenue, # actual
                                 positive = "1",
                                 mode = "everything")
DT_train_conf

## Testing Performance
base.tepreds <- predict(object = FD.rpart,
                        newdata = test, 
                        type = "class")


DT_test_conf <- confusionMatrix(data = base.tepreds, # predictions
                                reference = test$Revenue, # actual
                                positive = "1",
                                mode = "everything")
DT_test_conf




##Grid Search ##kval
grids <- expand.grid(cp = seq(from = 0,
                              to = 0.15,
                              by = 0.005))
ctrl_DT <- trainControl(method = "repeatedcv",
                        number = 5, # k = 10 folds
                        repeats = 3, # repeated CV 3 times
                        search = "grid"
                        ) 
set.seed(1112)
DTFit <- train(form = Revenue ~ ., 
               data = train[ ,c(vars, "Revenue")], 
               method = "rpart",
               trControl = ctrl_DT, 
               tuneGrid = grids)
DTFit


###Undersample ##Grid Search ##kval
grids <- expand.grid(cp = seq(from = 0,
                             to = 0.15,
                             by = 0.005))
ctrl_DT <- trainControl(method = "repeatedcv",
                        number = 5, # k = 10 folds
                        repeats = 3, # repeated CV 3 times
                        search = "grid",
                        sampling = "down") 
set.seed(1112)
DTFit <- train(form = Revenue ~ ., 
               data = train[ ,c(vars, "Revenue")], 
               method = "rpart",
               trControl = ctrl_DT) 
               #tuneGrid = grids)
DTFit

##Training Performance
outpreds <- predict(object = DTFit, 
                    newdata = train,)

train2 <- confusionMatrix(data = outpreds, 
                               reference = train$Revenue, 
                               positive = "1",
                               mode = "everything")
train2


##Testing Performance
outpreds <- predict(object = DTFit, 
                    newdata = test)

test2 <- confusionMatrix(data = outpreds, 
                               reference = test$Revenue, 
                               positive = "1",
                               mode = "everything")
test2

##Compare Testing and Performance
cbind(Training = train2$overall,
      Testing = test2$overall)



###OverSample ##Grid Search ##kval
grids <- expand.grid(cp = seq(from = 0,
                              to = 0.15,
                              by = 0.005))
ctrl_DT <- trainControl(method = "repeatedcv",
                        number = 5, # k = 10 folds
                        repeats = 3, # repeated CV 3 times
                        search = "grid",
                        sampling = "up") 
set.seed(1112)
DTFit <- train(form = Revenue ~ ., 
               data = train[ ,c(vars, "Revenue")], 
               method = "rpart",
               trControl = ctrl_DT, 
               tuneGrid = grids)
DTFit

##Training Performance
outpreds <- predict(object = DTFit, 
                    newdata = train)

train3 <- confusionMatrix(data = outpreds, 
                          reference = train$Revenue, 
                          positive = "1",
                          mode = "everything")
train3


##Testing Performance
outpreds <- predict(object = DTFit, 
                    newdata = test)

test3 <- confusionMatrix(data = outpreds, 
                         reference = test$Revenue, 
                         positive = "1",
                         mode = "everything")
test3

##Compare Testing and Performance
cbind(Training = train3$overall,
      Testing = test3$overall)






###Weights ##kval
target_var <- train$Revenue # identify target variable
weights <- c(sum(table(target_var))/(nlevels(target_var)*table(target_var)))
wghts <- weights[match(x = target_var, 
                       table = names(weights))]
ctrl_DT <- trainControl(method = "repeatedcv",
                        number = 5, # k = 10 folds
                        repeats = 3, # repeated CV 3 times
                        search = "grid")

set.seed(1112) # initialize random seed

DTFit_cw <- train(x = train[ ,vars],
                  y = train$Revenue,
                  method = "rpart", # use the rpart package
                  trControl = ctrl_DT, # control object
                  tuneLength = 5,   # try 5 cp values
                  weights = wghts) 
DTFit_cw

 
##Training Performance
outpreds <- predict(object = DTFit_cw, 
                    newdata = train)

train4 <- confusionMatrix(data = outpreds, 
                          reference = train$Revenue, 
                          positive = "1",
                          mode = "everything")
train4


##Testing Performance
outpreds <- predict(object = DTFit_cw, 
                    newdata = test)

test4 <- confusionMatrix(data = outpreds, 
                         reference = test$Revenue, 
                         positive = "1",
                         mode = "everything")
test4

##Compare Testing and Performance
cbind(Training = train4$overall,
      Testing = test4$overall)




###Oversample
train_up <- upSample(x = train[ ,vars], # predictors
                     y = train$Revenue, # target
                     yname = "Revenue") # name of y variable
plot(train$Revenue, main = "Original")
plot(train_up$Revenue, main = "RUP")
par(mfrow = c(1, 1)) 
FD.rpart <- rpart(formula = Revenue ~ ., 
                  data = train_up,
                  method = "class")

FD.rpart
FD.rpart$variable.importance

prp(x = FD.rpart,
    extra = 2) 




save.image(file = "decisiontree_Finalproject.RData")
