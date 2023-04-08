library(glmnet)

df <- read.csv("breastcancer.csv")

x <- as.matrix(df[, 1:1000])
y <- as.vector(df[, 1001])
cv <- cv.glmnet(x, y, family = "binomial")
cv2 <- cv.glmnet(x, y, family = "binomial", type.measure = "class")
par(mfrow = c(1, 2))
plot(cv)
plot(cv2)