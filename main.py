from models import models
#import ksvm_test

paths = ['./Data3']
sector = [0]

for path in paths:
	print("\n|-------------------------------------------------------------------------------|")
	print("\nRunning for: "+ path+":")
	print("\n|-------------------------------------------------------------------------------|")
	for sector_id in sector:
		run_models = models(path)
		
		#print("|-------------------------------------------------------------------------------|")
    	

		run_models.run_DNN(run_models.x_train,run_models.y_train,run_models.x_test,run_models.y_test,sector_id)
		#run_models.run_Logistic_regression(run_models.x_train,run_models.y_train,run_models.x_test,run_models.y_test,sector_id)
		#run_models.run_Linear_SVM(run_models.x_train,run_models.y_train,run_models.x_test,run_models.y_test)




