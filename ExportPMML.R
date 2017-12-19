#Import Data
Wine.data  <- read.csv("/Users/damienmarque/Desktop/Stage/Script_R/winequality-red.csv", header = TRUE, sep = ";")


#On ramène la variable à expliquer dans la première colonne.
Wine.data <- cbind(Wine.data[,c(12,1:11)]) 


#Nombre d'observations
nbr.obs <- 1599


#Define Test and Learning Table
Data.app <- Wine.data[1:(nbr.obs/3),1:12]
write.table(Data.app, "/Users/damienmarque/Anaconda/AnacondaProjects/Data_app.csv", row.names=FALSE, sep=";")
Data.app.y <- Wine.data[1:(nbr.obs/3),1]
write.table(Data.app.y, "/Users/damienmarque/Anaconda/AnacondaProjects/Data_app_y.csv", row.names=FALSE, sep=";")
Data.ver <- Wine.data[((nbr.obs/3)+1):((2*nbr.obs)/3),1:12]
write.table(Data.ver, "/Users/damienmarque/Anaconda/AnacondaProjects/Data_ver.csv", row.names=FALSE, sep=";")
Data.ver.y <- Wine.data[((nbr.obs/3)+1):((2*nbr.obs)/3),1]
write.table(Data.ver.y, "/Users/damienmarque/Anaconda/AnacondaProjects/Data_ver_y.csv", row.names=FALSE, sep=";")
Data.test <- Wine.data[(((2*nbr.obs)/3)+1):nbr.obs,2:12]
write.table(Data.test, "/Users/damienmarque/Anaconda/AnacondaProjects/Data_test.csv", row.names=FALSE, sep=";")
Data.test.y <- Wine.data[(((2*nbr.obs)/3)+1):nbr.obs,1]
write.table(Data.app, "/Users/damienmarque/Anaconda/AnacondaProjects/Data_app.csv", row.names=FALSE, sep=";")


#Define forestObject algorithm
forestObject <- randomForest(Data.app.y~.,data=Data.app,importance=TRUE,proximity=TRUE)


#Define Neuralnetwork algorithm
neural.netObject <- nnet(Data.app.y~., data=Data.app, size=2, decay=1, maxit=100)


#On transporte en Pmml
forestenpmml <- pmml(forestObject, model.name="randomForest_Model")
Neuralenpmml <- pmml(neural.netObject, model.name="neuralnetwork_Model")


#On enregistre nos résultats dans le fichier Pmml
saveXML(forestenpmml, "/Users/damienmarque/Desktop/Stage/Pmml/forestenpmml.pmml")
saveXML(Neuralenpmml, "/Users/damienmarque/Desktop/Stage/Pmml/Neuralenpmml.pmml")
