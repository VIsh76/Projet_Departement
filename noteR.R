#Ouverture des deux csv dans les data.frame dico et data
data=read.csv("Data_M.csv",header=TRUE)
dico=read.csv("Dico_M.csv",header=TRUE)
#On construit le data.frame qui ne retient que les variables dont le nom contient "Indusrtie"
Test=data[,grep("Industrie",dico$def)+2]
#On supprime la série 1656082 qui est celle qu'on veut expliquée
Test$"X001656082"=NULL
X=data[1:35,3:ncol(data)]
V=abs(cor(X,X$"X001656082"))
val=c(V)
names=c(rownames(V))
v=data.frame(val,row.names=names)
v=v[order(-v$val), , drop = FALSE]
val=v[1:10,1]
names=rownames(v)[1:10]
Res=data.frame(val,row.names=names)
Res2=X[rownames(Res)]
Res2$"X001656082"=NULL
Exp=data.matrix(Res2)
Var=X$"X001656082"
model=lm(Var~Exp)
S=summary(model)
anova(model)

Fail_Prevision(serie="X001656082",data,dico)