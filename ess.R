library(psych)
library(corrplot)
library("psych")
library(ggplot2)
library(car)

url <- "https://files-ms3lk4bu7-syrkis1.vercel.app"
data_survey <- read.csv(url, sep = ",")


describe(data_survey)
dim(data_survey)
head(data_survey)
head(data_survey[, c()])
