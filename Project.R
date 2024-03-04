library(class) #KNN
library(caret) #K-Fold Cross Validation
library(e1071) #SVM

#Update missing values:
dat <- Dataset
dat[88,13] <- 3
dat[167,12] <- 0
dat[193,12] <- 0
dat[267,13] <- 3
dat[288,12] <- 0
dat[303,12] <- 0

#Change data to be numerical:
dat <- sapply(dat, as.numeric)
dat <- as.data.frame(dat)
dat[, 'V14'] <- as.factor(dat[,'V14'])

#KNN:
#Splitting and scaling the data:
n <- nrow(dat)
RNGkind(sample.kind = "Rounding")
set.seed(1)
train <- sample(1:n, 0.8*n)

X.train <- scale(dat[train,-14])
X.test <- scale(dat[-train,-14],
                center = attr(X.train, "scaled:center"),
                scale = attr(X.train, "scaled:scale"))
Y.train <- dat$V14[train] 
Y.test <- dat$V14[-train]

#Validation Set:
knn.test.err <- numeric(101)

set.seed(1)
for(K in c(1:101)){
  knn.test <- knn(train = X.train,
                  test = X.test,
                  cl = Y.train,
                  k=K)
  knn.test.err[K] <- mean(knn.test != Y.test)
}

min(knn.test.err)
which.min(knn.test.err)
knn.test.err

plot(c(1:101), knn.test.err,
     type='b',
     xlab="K",
     ylab="Test error")

set.seed(1)
knn.pred <- knn(train=X.train,
                test=X.test,
                cl=Y.train,
                k=3)
summary(knn.pred)
mean(knn.pred != Y.test)
table(knn.pred, Y.test)

set.seed(1)
knn.pred <- knn(train=X.train,
                test=X.test,
                cl=Y.train,
                k=9)
summary(knn.pred)
mean(knn.pred != Y.test)
table(knn.pred, Y.test)

set.seed(1)
knn.pred <- knn(train=X.train,
                test=X.test,
                cl=Y.train,
                k=12)
summary(knn.pred)
mean(knn.pred != Y.test)
table(knn.pred, Y.test)

set.seed(1)
knn.pred <- knn(train=X.train,
                test=X.test,
                cl=Y.train,
                k=14)
summary(knn.pred)
mean(knn.pred != Y.test)
table(knn.pred, Y.test)

set.seed(1)
knn.pred <- knn(train=X.train,
                test=X.test,
                cl=Y.train,
                k=17)
summary(knn.pred)
mean(knn.pred != Y.test)
table(knn.pred, Y.test)

set.seed(1)
knn.pred <- knn(train=X.train,
                test=X.test,
                cl=Y.train,
                k=25)
summary(knn.pred)
mean(knn.pred != Y.test)
table(knn.pred, Y.test)
#LOOCV:
summary(dat$V14)
x.train <- scale(dat[,-14])
x.test <- scale(dat[,-14])
y.train <- dat$V14
y.test <- dat$V14

knn.loocv.err <- numeric(30)
set.seed(1)
for(K in c(1:30)){
  knn.cv.test <- knn.cv(train = x.train,
                        cl = dat$V14,
                        k=K)
  knn.loocv.err[K] <- mean(knn.cv.test != dat$V14)
}

min(knn.loocv.err)
which.min(knn.loocv.err)
knn.loocv.err

plot(c(1:30), knn.loocv.err,
     type='b',
     xlab="K",
     ylab="Test error")

set.seed(1)
knn.cv.pred <- knn.cv(train = x.train,
                      cl = dat$V14,
                      k=5)
summary(knn.cv.pred)
mean(knn.cv.pred != dat$V14)
table(knn.cv.pred, dat$V14)

set.seed(1)
knn.cv.pred <- knn.cv(train = x.train,
                      cl = dat$V14,
                      k=10)
summary(knn.cv.pred)
mean(knn.cv.pred != dat$V14)
table(knn.cv.pred, dat$V14)

set.seed(1)
knn.cv.pred <- knn.cv(train = x.train,
                      cl = dat$V14,
                      k=15)
summary(knn.cv.pred)
mean(knn.cv.pred != dat$V14)
table(knn.cv.pred, dat$V14)

set.seed(1)
knn.cv.pred <- knn.cv(train = x.train,
                      cl = dat$V14,
                      k=16)
summary(knn.cv.pred)
mean(knn.cv.pred != dat$V14)
table(knn.cv.pred, dat$V14)

set.seed(1)
knn.cv.pred <- knn.cv(train = x.train,
                      cl = dat$V14,
                      k=21)
summary(knn.cv.pred)
mean(knn.cv.pred != dat$V14)
table(knn.cv.pred, dat$V14)

set.seed(1)
knn.cv.pred <- knn.cv(train = x.train,
                      cl = dat$V14,
                      k=25)
summary(knn.cv.pred)
mean(knn.cv.pred != dat$V14)
table(knn.cv.pred, dat$V14)

#K-Fold Cross Validation
training <- dat[train,]
testing <- dat[-train,]

set.seed(1)
ctrl <- trainControl(method="cv", number = 5)
fit.knn <- train(V14 ~ ., data = training,
                 method = "knn",
                 trControl = ctrl,
                 tuneGrid = expand.grid(k = c(1,5,7,11,15,21)),
                 preProcess = c("center","scale"))
fit.knn
fit.knn.pred <- predict(fit.knn, testing)
mean(fit.knn.pred != dat$V14[-train])
table(fit.knn.pred, dat$V14[-train])

set.seed(1)
ctrl <- trainControl(method="cv", number = 10)
fit.knn <- train(V14 ~ ., data = training,
                 method = "knn",
                 trControl = ctrl,
                 tuneGrid = expand.grid(k = c(1,5,7,11,15,21)),
                 preProcess = c("center","scale"))
fit.knn
fit.knn.pred <- predict(fit.knn, testing)
mean(fit.knn.pred != dat$V14[-train])
table(fit.knn.pred, dat$V14[-train])

set.seed(1)
ctrl <- trainControl(method="cv", number = 15)
fit.knn <- train(V14 ~ ., data = training,
                 method = "knn",
                 trControl = ctrl,
                 tuneGrid = expand.grid(k = c(1,5,7,11,15,21)),
                 preProcess = c("center","scale"))
fit.knn
fit.knn.pred <- predict(fit.knn, testing)
mean(fit.knn.pred != dat$V14[-train])
table(fit.knn.pred, dat$V14[-train])

set.seed(1)
ctrl <- trainControl(method="cv", number = 20)
fit.knn <- train(V14 ~ ., data = training,
                 method = "knn",
                 trControl = ctrl,
                 tuneGrid = expand.grid(k = c(1,5,7,11,15,21)),
                 preProcess = c("center","scale"))
fit.knn
fit.knn.pred <- predict(fit.knn, testing)
mean(fit.knn.pred != dat$V14[-train])
table(fit.knn.pred, dat$V14[-train])

#Support Vector Classifier
#Cross-Validation
set.seed(1)
tune.linear <- tune(METHOD=svm,
                    V14~., data = dat[train,],
                    kernel = "linear",
                    ranges = list(cost=c(0.001,0.01,0.1,1,5,10,100,1000)))
summary(tune.linear)
bestmod = tune.linear$best.model
summary(bestmod)
ypred <- predict(tune.linear$best.model, newdata = dat[-train,])
table(predict=ypred, truth=ypred)

#Support Vector Machine
set.seed(1)
tune.poly <- tune(METHOD=svm,
                  V14~., data = dat[train,],
                  kernel="polynomial",
                  ranges = list(cost=c(0.001,0.01,0.1,1,5,10,100,1000),
                                degree=c(2,3,4,5)))
summary(tune.poly)
bestmod = tune.poly$best.model
summary(bestmod)
svmpoly <- predict(tune.poly$best.model, newdata = dat[-train,])
table(pred=svmpoly,true=dat[-train,"V14"])

set.seed(1)
tune.radial <- tune(METHOD=svm,
                    V14~., data = dat[train,],
                    kernel = "radial",
                    ranges = list(cost=c(0.001,0.01,0.1,1,5,10,100,1000),
                                  gamma=c(0.5,1,2,3,4)))
summary(tune.radial)
bestmod = tune.radial$best.model
summary(bestmod)
svmradial <- predict(tune.radial$best.model, newdata = dat[-train,])
table(pred=svmradial,true=dat[-train,"V14"])

set.seed(1)
svmfit <- svm(V14 ~ ., data=dat,
              kernel = "linear",
              cost = 0.1,
              scale = TRUE)
svmfit$index #Support Vectors
summary(svmfit)
table(svmfit$fitted, dat$V14)
