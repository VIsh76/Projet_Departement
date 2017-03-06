#Ouverture des deux csv dans les data.frame dico et data
data=read.csv("Data_M.csv",header=TRUE)
dico=read.csv("Dico_M.csv",header=TRUE)

Fail_Prevision  = function(serie,data,dico)
{
#On retire les trois derniers mois pour enlever la majeur partie des valeurs NA
X=data[1:35,3:ncol(data)]
Var=X[serie]
#On calcule les corrélations avec la série
V=abs(cor(X,Var))

#On les classe par ordres décroissant
val=c(V)
names=c(rownames(V))
v=data.frame(val,row.names=names)
v=v[order(-v$val), , drop = FALSE]
val=v[1:10,1]
names=rownames(v)[1:10]
Res=data.frame(val,row.names=names)

#On récupère les valeurs associées et on enleve la série (qui est correlé à 1 avec elle même)
Res2=X[rownames(Res)]
Res2[serie]=NULL

#On fait la régression linéaire
Exp=data.matrix(Res2)
Var=data.matrix(Var)
model=lm(Var~Exp)

return(model)
}

Fail_Prevision_decalage  = function(serie,decal,data,dico)
{
#On retire les trois derniers mois pour enlever la majeur partie des valeurs NA
X=data[1:(35-decal),3:ncol(data)]
Var=data[,serie]
Var=Var[(1+decal):35]
#On calcule les corrélations avec la série
V=abs(cor(X,Var))

#On les classe par ordres décroissant
val=c(V)
names=c(rownames(V))
v=data.frame(val,row.names=names)
v=v[order(-v$val), , drop = FALSE]
val=v[1:10,1]
names=rownames(v)[1:10]
Res=data.frame(val,row.names=names)

#On récupère les valeurs associées et on enleve la série (qui est correlé à 1 avec elle même)
Res2=X[rownames(Res)]
Res2[serie]=NULL

#On fait la régression linéaire
Exp=data.matrix(Res2)
Var=data.matrix(Var)
model=lm(Var~Exp)

return(model)
}

#Industrie
modelInd=Fail_Prevision(serie="X001656082",data,dico)
#Agriculture
modelAgr=Fail_Prevision(serie="X001656081",data,dico)

#Industrie avec decalage
modelInd2=Fail_Prevision_decalage(serie="X001656082",2,data,dico)


