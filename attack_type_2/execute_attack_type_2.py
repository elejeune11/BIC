import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, DotProduct
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn import svm

##########################################################################################
# learning false positives -- 
# training data: 10-10000 data points
# test data: -- BIC test dataset 
# attack data: 10-10000 data points, pulled from a different part of the training data
# -- AUC curve that shows a nn's ability to predict false positives in the orig mm
##########################################################################################
# import data / set up attack data
##########################################################################################
train_input_all = np.loadtxt('../BIC1/train_input_data.txt')
train_output_all =  np.loadtxt('../BIC1/train_output_data.txt')
num_train = train_input_all.shape[0]

# --> define the attack data (another part of the training data)
training_max = 1000
attack_input_all = train_input_all[training_max:,:]
attack_output_all = train_output_all[training_max:]

test_input = np.loadtxt('../BIC1/test_input_data.txt')
test_output = np.loadtxt('../BIC1/test_output_data.txt')
num_test = test_input.shape[0]

##########################################################################################
# import trained model
##########################################################################################
# --> need to run create_ROC_curves.py to make and save the metamodels first 
num_train_metamodel = 1000
filename = '../metamodels/pickled_gpc_%itrain.sav'%(num_train_metamodel)
clf = pickle.load( open(filename,'rb'))

# --> run metamodel --> --> -->
# --> make predictions on the different parts of the dataset -->
clf_pred_train = clf.predict(train_input_all[0:num_train_metamodel,:])

clf_pred_attack = clf.predict(attack_input_all)

clf_pred_test = clf.predict(test_input)

##########################################################################################
# identify falsely stable in the training data, attack data, and test data 
##########################################################################################
# 1 if falsely stable, 0 otherwise 
# set up empty arrays
fs_train = np.zeros((num_train_metamodel)) 
fs_attack = np.zeros((attack_input_all.shape[0]))
fs_test = np.zeros((num_test))

# compare attack prediction to ground truth (1 if falsely stable, 0 otherwise)
# --> train
for kk in range(0,num_train_metamodel):
	if train_output_all[kk] == 1 and clf_pred_train[kk] == 0: # ground truth is unstable, meta-model is stable
		fs_train[kk] = 1

# --> attack
for kk in range(0,attack_input_all.shape[0]):
	if attack_output_all[kk] == 1 and clf_pred_attack[kk] == 0: # ground truth is unstable, meta-model is stable
		fs_attack[kk] = 1 
	
# --> test 
for kk in range(0,num_test):
	if test_output[kk] == 1 and clf_pred_test[kk] == 0: # ground truth is unstable, meta-model is stable
		fs_test[kk] = 1

##########################################################################################
# train a model to find the falsely stable
##########################################################################################
num_attack_list = [100, 1000, 10000] # -- vary the # of points allowed in the attack model

plt.figure()


for jjj in range(0,len(num_attack_list)):
	num_attack = num_attack_list[jjj]
	# --> NN attack model 
	clf_nn_attack = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(200,200,200), random_state=1)
	clf_nn_attack.fit(attack_input_all[0:num_attack,:], fs_attack[0:num_attack] )

	# --> evaluate NN attack model 
	y_score_nn_attack = clf_nn_attack.predict_proba(test_input)[:,1]
	fpr_nn, tpr_nn, _  = roc_curve(fs_test, y_score_nn_attack)
	roc_auc_nn = auc(fpr_nn, tpr_nn)
	
	plt.plot(fpr_nn, tpr_nn,label='%i attack, %.3f'%(num_attack,roc_auc_nn))
		
	np.savetxt( 'attack2_fpr_nn_%itrain_%iattack.txt'%(num_train_metamodel,num_attack) , fpr_nn)
	np.savetxt( 'attack2_tpr_nn_%itrain_%iattack.txt'%(num_train_metamodel,num_attack) , tpr_nn)

##########################################################################################
# ugly plot of ROC curves 
##########################################################################################	
plt.plot([0,1],[0,1],'k--') 
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('ROC Attack 2 NN %i'%(num_train_metamodel))
plt.savefig('ROC_NN_attack2_test_%i'%(num_train_metamodel))