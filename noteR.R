#Ouverture des deux csv dans les data.frame dico et data
data=read.csv("Data_M.csv",header=TRUE)
dico=read.csv("Dico_M.csv",header=TRUE)
#On construit le data.frame qui ne retient que les variables dont le nom contient "Indusrtie"
Test=data[,grep("Industrie",dico$def)+2]
#On supprime la série 1656082 qui est celle qu'on veut expliquée
Test$"X001656082"=NULL
