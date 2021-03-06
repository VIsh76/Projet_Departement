#Ouverture des deux csv dans les data.frame dico et data
data=read.csv("Data_M.csv",header=TRUE)
dico=read.csv("Dico_M.csv",header=TRUE)

Fail_Prevision  = function(serie,data,dico)
{
#On retire les trois derniers mois pour enlever la majeur partie des valeurs NA
X=data[1:35,3:ncol(data)]
Var=X[serie]
#On calcule les corr�lations avec la s�rie
V=abs(cor(X,Var))

#On les classe par ordres d�croissant
val=c(V)
names=c(rownames(V))
v=data.frame(val,row.names=names)
v=v[order(-v$val), , drop = FALSE]
val=v[1:10,1]
names=rownames(v)[1:10]
Res=data.frame(val,row.names=names)

#On r�cup�re les valeurs associ�es et on enleve la s�rie (qui est correl� � 1 avec elle m�me)
Res2=X[rownames(Res)]
Res2[serie]=NULL



#On fait la r�gression lin�aire
Exp=data.matrix(scale(Res2))
Var=data.matrix(Var)
model=lm(Var~Exp)

return(model)
}

Fail_Prevision_decalage  = function(serie,decal,nbv,data,dico)
{
#On retire les trois derniers mois pour enlever la majeur partie des valeurs NA
X=data[1:(35-decal),3:ncol(data)]
Var=data[,serie]
Var=Var[(1+decal):35]
#On calcule les corr�lations avec la s�rie
V=abs(cor(X,Var))

#On les classe par ordres d�croissant
val=c(V)
names=c(rownames(V))
v=data.frame(val,row.names=names)
v=v[order(-v$val), , drop = FALSE]
val=v[1:nbv+1,1]
names=rownames(v)[1:nbv+1]
Res=data.frame(val,row.names=names)

#On r�cup�re les valeurs associ�es et on enleve la s�rie (qui est correl� � 1 avec elle m�me)
Res2=X[rownames(Res)]
Res2[serie]=NULL

#On fait la r�gression lin�aire
Exp=data.matrix(scale(Res2))
Var=data.matrix(Var)
model=lm(Var~Exp)

return(model)
}

readDico1 = function(serie)
{
s=unlist(strsplit(serie,"ExpX00"))
dico[which(dico$IDBANK==s[2]),]
}

ReadDico1 = function(Lserie)
{
for ( i in 1:nrow(Lserie)) { 
print(readDico1(Z[i,1]))
}
}

readDico2 = function(serie)
{
s=unlist(strsplit(serie,"X00"))
dico[which(dico$IDBANK==s[2]),]
}

ReadDico2 = function(Lserie)
{
for ( i in 1:nrow(Lserie)) { 
print(readDico2(Z[i,1]))
}
}

gain.info = function(serie,data,dico)
{
library(mlr)
library(FSelector)
X=data[1:35,3:ncol(data)]
Var=X["X001656082"]
#On calcule les corr�lations avec la s�rie
V=abs(cor(X,Var))

#On les classe par ordres d�croissant
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
M=U$name[1:10]
ReadDico2(data.frame(M))
}

#Industrie
modelInd=Fail_Prevision(serie="X001656082",data,dico)
#Agriculture
modelAgr=Fail_Prevision(serie="X001656081",data,dico)

#Industrie avec decalage
modelInd2=Fail_Prevision_decalage(serie="X001656082",2,10,data,dico)