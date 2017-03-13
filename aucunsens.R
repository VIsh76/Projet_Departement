install.packages("FSelector")
library(FSelector)


val=c(V)
names=c(rownames(V))
v=data.frame(val,row.names=names)
v=v[order(-v$val), , drop = FALSE]
val=v[1:10,1]
names=rownames(v)[1:10]
Res=data.frame(val,row.names=names)


X=data[1:35,3:ncol(data)]
Var=X["X001656082"]
#On calcule les corrélations avec la série
V=abs(cor(X,Var))

#On les classe par ordres décroissant
val=c(V)
names=c(rownames(V))
v=data.frame(val,row.names=names)
v=v[order(-v$val), , drop = FALSE]
val=v[1:1000,1]
names=rownames(v)[1:1000]
Res=data.frame(val,row.names=names)

Res2=X[rownames(Res)]

P.task=makeClassifTask(data = Res2, target = "X001656082")
u=generateFilterValuesData(P.task, method = "information.gain")
U=u$data
U=U[order(-U$information.gain), , drop = FALSE]
U$name
write(U$name,"test.txt")

Res2[serie]=NULL

#ensure results are repeatable
set.seed(7)
# load the library
library(mlbench)
library(caret)
# load the dataset
data(PimaIndiansDiabetes)
# prepare training scheme
control <- trainControl(method="repeatedcv", number=10, repeats=3)
# train the model
model <- train(X001694104~., data=P, method="gbm", preProcess="scale", trControl=control)
# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)

control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
results <- rfe(PimaIndiansDiabetes[,1:8], PimaIndiansDiabetes[,9], sizes=c(1:8), rfeControl=control)
# summarize the results
print(results)
# list the chosen features
predictors(results)
# plot the results
plot(results, type=c("g", "o"))


a="X001656082"
v=unlist(strsplit(a,"X00"))
v[2]
b=dico$IDBANK==v[2]
which(b)
dico[which(b),]
Z=data.frame(U$name)
Z=within(Z,{U.name <- as.character(U.name)})
within(Z,{U.name <- readDico(U.name)})

for ( i in 1:nrow(Z)) { 
print(readDico(Z[i,1]))
}