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
library(cluster) # clustering
library(factoextra) # cluster validation
library(fpc) # kmeans cluster plots
library(Rtsne) # dimension reduction
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

#######Get rid of near zero variance features - Special day 
nearZeroVar(df, names = TRUE)
NZV<- nearZeroVar(df, saveMetrics = TRUE)
#var(onlynums$BounceRates)
#var(FD_bcs$SpecialDay)

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

nums <- c('Administrative','Administrative_Duration','Informational',	'Informational_Duration',
          'ProductRelated',	'ProductRelated_Duration',	'BounceRates',	'ExitRates',	
          'PageValues')

###Get rid of high correlation 
cor_vars <- cor(x = df[ ,nums])
symnum(x = cor_vars,
       corr = TRUE)
high_corrs <- findCorrelation(x = cor_vars, 
                              cutoff = .80, 
                              names = TRUE)
nums <- nums[!nums %in% high_corrs]

vars <- c(noms, ords, nums)  #combine vectors
set.seed(1112) 

#Standardize the data - 0 values present use yeo
cen_bcs <- preProcess(x = df[ ,vars], 
                      method = c("YeoJohnson", "center", "scale"))
FD_bcs <- predict(object = cen_bcs,
                  newdata = df)


hdist <- daisy(x = FD_bcs[, -18], # omit Revenue
               metric = "gower")
summary(hdist)


# Apply Centroid-based Clustering
cen <- hclust(d = hdist ^ 2, 
              method = "centroid")
# Plot the dendrogram
plot(cen, 
     sub = NA, xlab = NA, 
     main = "Centroid Linkage")
# Overlay boxes identifying clusters
# for a k = 2 clustering solution
rect.hclust(tree = cen, 
            k = 2, 
            border = hcl.colors(2))

# Create a vector of cluster assignments
cen_clusters <- cutree(tree = cen, 
                       k = 2)


##### Apply Ward's Clustering
wards <- hclust(d = hdist, 
                method = "ward.D2")
# Plot the dendrogram
plot(wards, 
     xlab = NA, sub = NA, 
     main = "Ward's Method")
# Overlay boxes identifying clusters
# for a k = 10 clustering solution
rect.hclust(tree = wards, 
            k = 10, 
            border = hcl.colors(10))
# Create a vector of cluster assignments
wards_clusters <- cutree(tree = wards, 
                         k = 10)

## Visualizing the Solutions
# We can plot the distance matrix in
# reduced dimensionality and compare groupings
# that we see to cluster assignments from HCA
# using the Rtsne() function in the Rtsne package
ld_dist <- Rtsne(X = hdist, 
                 is_distance = TRUE)

# The 2 dimensional representation is in the
# Y list component of ld_dist. We will coerce
# it to a dataframe
lddf_dist <- data.frame(ld_dist$Y)

# Ward's Method
ggplot(data = lddf_dist, 
       mapping = aes(x = X1, y = X2)) +
  geom_point(aes(color = factor(wards_clusters))) +
  labs(color = "Cluster")


set.seed(1112)

kmeans1 <- kmeans(x = FD_bcs[ ,nums], 
                  centers = 4, 
                  trace = FALSE, 
                  nstart = 30)
kmeans1
# We can view the frequency distribution 
# of the clusters
kmeans1$size


## Visualize kMC Clusters
# We can use the fviz_cluster() function
# from the factoextra package to visualize
# the kMC cluster solution
fviz_cluster(object = kmeans1, 
             data = FD_bcs[ ,nums])

## Alternatively, we can plot the cluster
# solution on top of an Rtsne dimension-
# reduced data

ggplot(data = lddf_dist, 
       mapping = aes(x = X1, y = X2)) +
  geom_point(aes(color = factor(kmeans1$cluster))) +
  labs(color = "Cluster")

## Describe kMC Clusters
clus_means_kMC <- aggregate(x = FD_bcs[ ,nums], 
                            by = list(kmeans1$cluster), 
                            FUN = mean)
clus_means_kMC

matplot(t(clus_means_kMC[ ,-1]), # Ignore cluster # column
        type = "l", # line plot ("l")
        ylab = "", # no y-axis label
        xlim = c(0, 7), # add space for legend
        xaxt = "n", # no x-axis
        col = 1:4, # 10 colors (for k = 4)
        lty = 1:4, # 10 line types (for k = 4)
        main = "Cluster Centers") # main title
# Add custom x-axis labels
axis(side = 1, # x-axis
     at = 1:7, # x values 1-7
     labels = nums, # variable names as labels
     las = 2) # flip text to vertical
legend("left", # left position
       legend = 1:4, # 10 lines
       col = 1:4, # 10 colors
       lty = 1:4, # 10 line types
       cex = 0.6) # reduce text size



set.seed(1112)

# We use the pam() function from the cluster
# package. We use the gower distance matrix
# (hdist) as input, set diss = TRUE since we
# are using a custom distance matrix
pam1 <- pam(x = hdist,
            diss = TRUE,
            k = 4)
pam1

# We can view the rows representing the 
# mediods, which can be use to exemplify 
# a cluster (in order from 1:k clusters)
df[pam1$medoids, ]

# We can plot the distance matrix in
# reduced dimensionality (lddf_dist, created
# in HCA example) and compare groupings
# that we see to cluster assignments from 
# pam

# Cluster membership for the data is in the
# clustering list component of the pam object
head(pam1$clustering)

# We can plot the 2-dimensional representation
# of the PAM cluster solution and use the
# cluster membership as the color

ggplot(data = lddf_dist, 
       mapping = aes(x = X1, y = X2)) +
  geom_point(aes(color = factor(pam1$clustering))) +
  labs(color = "Cluster")
aggregate(x = df[ ,c(ords,noms)], 
          by = list(pam1$clustering), 
          FUN = table)


#Numerical Variables
clus_means_PAM <- aggregate(x = FD_bcs[ ,nums], 
                            by = list(pam1$clustering), 
                            FUN = mean)
clus_means_PAM

# We can use the matplot() function to
# visualize the (scaled) cluster centers
# to observe differences
matplot(t(clus_means_PAM[,-1]), # Ignore cluster # column
        type = "l", # line plot ("l")
        ylab = "", # no y-axis label
        xlim = c(0, 7), # add space for legend
        xaxt = "n", # no x-axis
        col = 1:4, # 4 colors (for k = 4)
        lty = 1:4, # 4 line types (for k = 4)
        main = "Cluster Centers") # main title
# Add custom x-axis labels
axis(side = 1, # x-axis
     at = 1:7, # x values 1-7
     labels = nums, # variable names as labels
     las = 2) # flip text to vertical
legend("left", # left position
       legend = 1:4, # 4 lines
       col = 1:4, # 4 colors
       lty = 1:4, # 4 line types
       cex = 0.6) # reduce text size




library(cluster) # clustering
library(fpc)
args(wss_plot)
# kMeans Clustering (method = "kmeans")
wss_plot(scaled_data = FD_bcs[ ,nums], # dataframe
         method = "kmeans", # kMC
         max.k = 15, # maximum k value
         seed_no = 1112) # seed value for set.seed()
# Elbow at k = 4 or 5 

# kMeans Clustering (kMC) (method = "kmeans")
sil_plot(scaled_data = FD_bcs[ ,nums], # scaled data
         method = "kmeans", # kMC
         max.k = 15, # maximum k value
         seed_no = 1112) # seed value for set.seed()
# k = 8 is max, k = 4,5 local maxima
save.image(file = "kmeans_Finalproject.RData")

