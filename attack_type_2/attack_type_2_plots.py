import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

##########################################################################################
# nice plot of ROC curves that show the performance of a type 2 attack
##########################################################################################	
# need to run ``execute_attack_type_2.py" script first 
##########################################################################################	

num_train_list = [100,1000, 10000] # can change this, this is just what's in the paper
num_attack_list = [100, 1000, 10000] # ^^ same

for num_train_metamodel in num_train_list:
	fig = plt.figure()
	plt.style.use('el_papers.mplstyle')
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	fig.set_figheight(7)
	fig.set_figwidth(7)
	
	color_list = [(1.0,.85,.85),(.75,.5,.5),(.75,0,0)] 
	idx = 0 
	
	for num_attack in num_attack_list:
		
		fpr_nn = np.loadtxt( 'attack2_fpr_nn_%itrain_%iattack.txt'%(num_train_metamodel,num_attack))
		tpr_nn = np.loadtxt( 'attack2_tpr_nn_%itrain_%iattack.txt'%(num_train_metamodel,num_attack))
		roc_auc_nn = auc(fpr_nn, tpr_nn)
		
		plt.plot(fpr_nn, tpr_nn,color=color_list[idx],label='%i pt attack, AUC=%.3f'%(num_attack,roc_auc_nn))
		idx += 1 
		
	plt.plot([0,1],[0,1],'k--') 

	plt.legend()
	plt.title('Attack 2 ROC, %i training points'%(num_train_metamodel))
	plt.xlabel('false positive rate')
	plt.ylabel('true positive rate')
	plt.tight_layout()

	plt.savefig('BIC1_attack2_ROC_%i_num_train'%(num_train_metamodel))
	plt.savefig('BIC1_attack2_ROC_%i_num_train.eps'%(num_train_metamodel))