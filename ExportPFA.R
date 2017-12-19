# you only need to run this once because it 
# installs the package
devtools::install_github('ReportMort/hadrian', ref='feature/add-r-package-structure', subdir='aurelius', quick=T, ) 

options(stringsAsFactors = FALSE)
library(aurelius);library(randomForest) 

# make sure column names dont have periods in them
iris2 <- iris
colnames(iris2) <- gsub('\\.', '_', colnames(iris2)) 

write_json(iris2, "/Users/damienmarque/Stage/Hadrian/iris2.json")

# fit a random forest model
rf_model <- randomForest(Species ~ ., data=iris2) 

# convert that model to PFA
# pred_type='prob' means that the output is the 
# probability of all classes. If you want 
# the majority vote, then specify pred_type='response'
# check all options by running ?pfa.randomForest
# python.exec("from titus.genpy import PFAEngine")
rf_model_as_pfa <- pfa(rf_model, pred_type='prob') 

# you can check the predictions from this model
# by first converting to a pfa_engine
pfa_model <- pfa_engine(rf_model_as_pfa) 

# confirm that the pfa engine produces the 
# same prediction as the "predict" method in R
# test_dat <- iris2[73,1:4]
test_dat <- iris2[83,1:4]
round(pfa_model$action(as.list(test_dat)), 6)
round(unclass(predict(object=rf_model, newdata=test_dat, type='prob')), 6) 


# you can export your model for use in other systems
write_pfa(rf_model_as_pfa, file = '~/my_rf_model.pfa')

