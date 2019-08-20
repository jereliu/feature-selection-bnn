library(magrittr)
library(ranger)
library(knockoff)
library(BNN)
library(bkmr)

setwd("~/Project/feature-selection-bnn")

y = read.csv("y_train.csv", header = FALSE) %>% as.matrix
X = read.csv("X_train.csv", header = FALSE) %>% as.matrix

dat = data.frame(X)
dat$y <- y

# knockoff
knockoff.filter(X, unlist(y),
                statistic = stat.glmnet_coefdiff,
                fdr = 0.5)$selected
knockoff.filter(X, unlist(y) ,
                statistic = stat.random_forest,
                fdr = 0.5)$selected

# random forest
res_rf <- ranger(y ~ ., data = dat, importance = "permutation")
res_rf <- importance_pvalues(res_rf, method="altmann",
                             formula = y ~ ., data = dat,
                             num.permutations = 100)[, "pvalue"]
res_rf <- which(res_rf <=  0.05)

# bnn
bnn_res <- BNNsel(X, y, train_num = as.integer(0.95 * length(y)), 
                  total_iteration = 5000)
res_bnn <- which(bnn_res$mar > 0.1)

# kernel
fitkm <- kmbayes(y = y, Z = X, iter = 1000, 
                 verbose = FALSE, varsel = TRUE)
res_bkmr <- ExtractPIPs(fitkm)
res_bkmr <- which(res_bkmr$PIP > 0.5)
