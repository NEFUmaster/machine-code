# Slightly modified from http://127.0.0.1:15296/library/kuenm/html/kuenm_occsplit.html

library(kuenm)

# set the working directory
setwd("C:/Users/R6meg/Desktop/tree")

# arguments
occs <- "hua_shu_joint.csv"
train_prop <- 0.75
method = "random"
name <- "hua_shu"

# running
data_split <- kuenm_occsplit(occ.file = occs, train.proportion = train_prop, method = method, name = "hua_shu"
)