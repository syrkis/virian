# imports
library("haven")
library(plyr)

# dataset files
setwd("~/virian/data/pew")
folders <- dir()
file <- paste(folders[1], list.files(path=folders[1], pattern='*.sav'), sep='/')
data <- read_sav(file)