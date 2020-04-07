import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, DotProduct
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import pickle
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
##########################################################################################
# ROC curves -- notes
# Accuracy = (1 - Error) = (TP + TN) / (PP + NP) = Pr(C) -- probability of a correct classification
# Sensitivity = TP / (TP + FN) = TP/PP 
# Specificity = TN / (TN + FP) = TN/NP
# ROC curve structure:
#		==> x axis is the false positive rate
#		==> y axis is the true positive rate
#		==> diagonal line shows a useless classifier 
##########################################################################################
###########################################################################################
# import data  -- can change directory location -- will have to update plot + filenames accordingly
train_input_all = np.loadtxt('../BIC1/train_input_data.txt')
train_output_all =  np.loadtxt('../BIC1/train_output_data.txt')
test_input = np.loadtxt('../BIC1/test_input_data.txt')
test_output = np.loadtxt('../BIC1/test_output_data.txt')

# --> ROC curves with more and more training data
#num_train_li = [10, 250, 500, 750, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000 ] 
num_train_li = [50]

for num_train in num_train_li:
	print('starting num_train:',num_train)
	train_input = train_input_all[0:num_train,:]
	train_output = train_output_all[0:num_train]

	##########################################################################################
	# create models --> SVM, NN, GPC 
	create_all_models = True

	# SVM model --> 
	create_svm_model = True and create_all_models
	if create_svm_model == True:
		# create SVM model 
		clf_svm = svm.SVC()
		clf_svm.fit(train_input, train_output)
		# save SVM model
		filename = 'pickled_svm_%itrain.sav'%(num_train)
		pickle.dump(clf_svm,open(filename,'wb'))
	else:
		# load SVM model 
		filename = 'pickled_svm_%itrain.sav'%(num_train)
		clf_svm = pickle.load( open(filename,'rb'))

	# NN model --> 
	create_nn_model = True and create_all_models
	if create_nn_model == True:
		# create NN model
		clf_nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(200,200,200), random_state=1)
		clf_nn.fit(train_input, train_output)
		# save NN model
		filename = 'pickled_nn_%itrain.sav'%(num_train)
		pickle.dump(clf_nn,open(filename,'wb'))
	else:
		# load NN model 
		filename = 'pickled_nn_%itrain.sav'%(num_train)
		clf_nn = pickle.load( open(filename,'rb'))

	# GPC model -->
	create_gpc_model = True and create_all_models
	if create_gpc_model == True:
		# create GPC model
		clf_gpc = GaussianProcessClassifier(kernel=1.0 * RBF(10.0))
		clf_gpc.fit(train_input, train_output)
		# save GPC model
		filename = 'pickled_gpc_%itrain.sav'%(num_train)
		pickle.dump(clf_gpc,open(filename,'wb'))
	else:
		# load GPC model
		filename = 'pickled_gpc_%itrain.sav'%(num_train)
		clf_gpc = pickle.load( open(filename,'rb'))
	
	##########################################################################################
	# create ROC data

	# --> for SVM:
	y_score_svm = clf_svm.decision_function(test_input)
	fpr_svm, tpr_svm, _  = roc_curve(test_output, y_score_svm)
	roc_auc_svm = auc(fpr_svm, tpr_svm)

	# --> for NN:
	y_score_nn = clf_nn.predict_proba(test_input)[:,1]
	fpr_nn, tpr_nn, _  = roc_curve(test_output, y_score_nn)
	roc_auc_nn = auc(fpr_nn, tpr_nn)

	# --> for NN:
	y_score_gpc = clf_gpc.predict_proba(test_input)[:,1]
	fpr_gpc, tpr_gpc, _  = roc_curve(test_output, y_score_gpc)
	roc_auc_gpc = auc(fpr_gpc, tpr_gpc)

	##########################################################################################
	# make nice plots
	fig = plt.figure()
	plt.style.use('el_papers.mplstyle')
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	fig.set_figheight(6)
	fig.set_figwidth(6)

	plt.plot(fpr_svm,tpr_svm,color=(.5,.25,.5),label='SVM, AUC=%0.3f'%(roc_auc_svm))
	plt.plot(fpr_nn,tpr_nn,color=(0,0,0.75),label='NN, AUC=%0.3f'%(roc_auc_nn))
	plt.plot(fpr_gpc,tpr_gpc,color=(.75,0,0),label='GPC, AUC=%0.3f'%(roc_auc_gpc))

	plt.plot([0,1],[0,1],'k--') 

	plt.legend()
	plt.title('BIC-1 ROC, %i training points'%(num_train))
	plt.xlabel('false positive rate')
	plt.ylabel('true positive rate')
	plt.tight_layout()

	plt.savefig('BIC1_ROC_%i_num_train'%(num_train))
	plt.savefig('BIC1_ROC_%i_num_train.eps'%(num_train))

##########################################################################################
# savetxt for use later / re-plotting 
# np.savetxt('fpr_svm_%i.txt'%(num_train),fpr_svm)
# np.savetxt('tpr_svm_%i.txt'%(num_train),tpr_svm)
# 
# np.savetxt('fpr_nn_%i.txt'%(num_train),fpr_nn)
# np.savetxt('tpr_nn_%i.txt'%(num_train),tpr_nn)
# 
# np.savetxt('fpr_gpc_%i.txt'%(num_train),fpr_gpc)
# np.savetxt('tpr_gpc_%i.txt'%(num_train),tpr_gpc)