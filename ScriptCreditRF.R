#######################################################
####################################################### Data Management 
#######################################################

#On Nettoie la base de donnée
Data.1 <-cbind(data_cv_pred_rf[,c(2,1,3:27)]) 
Data.1 <-cbind(Data.1[,c(1:22,25:27,24,23)])
#On affiche le résumé de la table.
Summary.1 <- summary(Data.1)
#On modifie deux features ici Statut_RC et NSC0064.
Data.1$STATUT_RC[Data.1$STATUT_RC==1]<-0 
Data.1$STATUT_RC[Data.1$STATUT_RC==2]<-1 
Summary.2 <- summary(Data.1$STATUT_RC)
Data.2 <- matrix(0,nrow=408253, ncol=3)
colnames(Data.2) <- c("NSCO064_1", "NSCO064_2", "NSCO064_3")
Data.2 <- as.data.frame(Data.2)
Data.2$NSCO064_1[Data.1$NSCO064==1]<-1                 
Data.2$NSCO064_2[Data.1$NSCO064==2]<-1                 
Data.2$NSCO064_3[Data.1$NSCO064==3]<-1 
Summary.3 <- summary(Data.2)
#On crée la base de données qu l'on va utilisée comme référence au cours du temps et qui s'appelle DataS. 
Data <- cbind(Data.1[,c(1:26)])                   
Data <- cbind.data.frame(Data,Data.2) 
Data$defaut <- factor(Data$defaut)
DataS <- cbind(Data[,c(1,3:29)])
#On cré une base d'apprentissage et de test. 
DataSTarget1 <- DataS[which(DataS$defaut=="1"),]
DataSTarget0 <- DataS[which(DataS$defaut=="0"),]
matrixDM<-function(x)
{
  DataSLearn <- matrix(nrow = x, ncol = dim(DataS)[2])
  VecteurAleatoire1 <- round(runif(2.6*x/100, min=0, max=dim(DataSTarget1)[1]), digits=0)
  VecteurAleatoire0 <- round(runif(97.4*x/100, min=0, max=dim(DataSTarget0)[1]), digits=0)
  DataSLearn <- rbind(DataS[VecteurAleatoire1,],DataS[VecteurAleatoire0,])
  return(DataSLearn)# il s'agit du rÃ©sultat que va renvoyer la fonction
}
BaseApprentissage <- matrixDM(300000)
BaseTest <- matrixDM(100000)









#######################################################
####################################################### Machine Learning
#######################################################

#On charge les librarys randomForest, parallel, doMC et crossval
library(randomForest)
library(parallel)
library(doMC)
library(crossval)
library(MVR)
#On cherche à connaître le nombre de coeur que l'on dispose.
detectCores()
detectCores(logical=FALSE)
#On définit à présent notre bombre de coeur sur lequel on travaille.
cl <- makeCluster(16)
registerDoMC(cl)

#On construit le modèle
Temps1 <-proc.time()
RandomForestModel.model <- randomForest(BaseApprentissage$defaut~.,BaseApprentissage,mtry=8,ntree=100)
Temps1 <- Temps1-proc.time()
#On affiche le vecteur err.rate qui contient pour chaque ntree lignes
#le taux d'erreur global OOB, le taux d'erreur OOB sur la première classe 
#et le taux d'erreur OOB sur la seconde classe.
tail(RandomForestModel.model$err.rate)
#On plot les informations précédentes.
plot(RandomForestModel.model$err.rate[,1], type='l', ylim=c(0,.1), xlab="nombre d'arbres", ylab='erreur OOB')
lines(RandomForestModel.model$BaseTest$err.rate[,1], type='l', lwd=2, col='red')
#Taux d'erreurs minimum en OOB
min.err.app <- min(RandomForestModel.model$err.rate[,"OOB"])
min.err.idx.app <- which(RandomForestModel.model$err.rate[,"OOB"]==min.err)
#Mesure l'importance des varaibles sur le critère du MeanDecreraseAccuracy
rf.imp.1 <- importance(RandomForestModel.model,type = 1)[order(importance(RandomForestModel.model,type = 1),decreasing=TRUE),]
#Mesure l'importance des varaibles sur le critère du MeanDecreraseGini
rf.imp.2 <- importance(RandomForestModel.model,type = 2)[order(importance(RandomForestModel.model,type = 2),decreasing=TRUE),]
#On affiche le fichier plot correspondant.
varImpPlot(RandomForestModel.model)
#Distribution de l'importance
sd(as.vector(importance(RandomForestModel.model, type=1, scale=T)))
sd(as.vector(importance(RandomForestModel.model, type=2, scale=T)))
#Une autre manière de représenter les résultats.
par(mar= c(8,4,4,0))
barplot(rf.imp.2, col=gray(0:nrow(RandomForestModel.model$importance)/nrow(RandomForestModel.model$importance)), ylab='importance',ylim = c(0,3000), cex.names=0.8, las=3)

#On va choisir à présent le nombre de mtry.
mtry <- tuneRF(DataS[,-1],DataS[,1],mtryStart=1, ntreeTry = 87, stepFactor = 2, improve = 0.001 )
best.mtry <- mtry[which.min(mtry[,2]),1]
best.mtry
plot(mtry)

set.seed(235)
nsimul <- 100
nvarmin <- 1
nvarmax <- 8
auc <-matrix(NA, nvarmax-nvarmin+1,2)
for(nvar in nvarmin:nvarmax)
{
  auc[nvar-nvarmin+1,1] <- nvar
  rft <- matrix(NA,nrow(BaseTest), nsimul+1)
  for(i in 1:nsimul)
  {
    rf <- randomForest(BaseApprentissage$defaut~.,BaseApprentissage,importance=F, ntree=87, mtry=nvar, replace=T, keep.forest=T, nodesize=5)
    rft[,i] <- predict(rf ,BaseTest, type='prob')[,2]
  }
  rft[,nsimul+1] <- apply(rft[,1:nsimul], 1, mean)
  pred <- prediction(rft[,nsimul+1], BaseTest$defaut, label.ordering=NULL)
  auc[nvar-nvarmin+1,2] <- performance(pred,"auc")@y.values[[1]]
}
colnames(auc) <- c("nb variables", "AUC test")

BaseTest$RandomForestModel.model <- predict(RandomForestModel.model, BaseTest, type='prob')
pred <- prediction(BaseTest$RandomForestModel.model, BaseTest$Cible, label.ordering=c(0,1))
performance(pred,"auc")@y.values[[1]]

# 5. Once the PMML library is uploaded, simply type pmml(Object) 
#    The PMML package reads the internal representation of the random forest object 
#    in R and generates a PMML 4.2 file from it 
library(pmml) 
PMML<-pmml(wine_rf, model.name="randomForest_Model", app.name="Rattle/PMML", 
           description="Random Forest Tree Model", copyright="Customer Analytics", transforms=NULL, 
           unknownValue=NULL)
# 6. Your PMML file is exported and ready for deployment.  
# It is very easy to generate PMML from R
nomFichier<-paste("/apps/projets/RISK-BDDF_SAGED/Damien","RandomForestLV", ".pmml", sep="") 
sink(nomFichier)  
print(RandomForest.ppml) 
sink()
