library(glmnet)
install.packages("ggfortify")
library(MASS)
library(survival)
load("LymphomaData.rda")
attach("LymphomaData.rda")



names(patient.data)

x <- t(patient.data$x)
y <- patient.data$time
delta <- patient.data$status
Surv(y, delta)


library(ranger)
library(ggplot2)
library(dplyr)
library(ggfortify)

cv.fit <- cv.glmnet(x, Surv(y, delta), family = "cox")
cv.fit$lambda.min
fit2 <- glmnet(x, Surv(y, delta), lambda = cv.fit$lambda.min, family = "cox")
b <- fit2$beta
z <- sign(drop(x %*% fit2$beta))
fit3 <- survfit(Surv(y, delta) ~ z)
autoplot(fit3)
summary(cv.fit)

mean(y[z == 1])

b[b != 0]

mean(y[z == -1])
