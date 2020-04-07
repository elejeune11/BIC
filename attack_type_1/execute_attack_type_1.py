import numpy as np
import pickle
import matplotlib.pyplot as plt
##########################################################################################
# import data 
##########################################################################################
train_input_all = np.loadtxt('../BIC1/train_input_data.txt')
train_output_all =  np.loadtxt('../BIC1/train_output_data.txt')
test_input = np.loadtxt('../BIC1/test_input_data.txt')
test_output = np.loadtxt('../BIC1/test_output_data.txt')

num_train = train_input_all.shape[0]
num_test = test_input.shape[0]
total_num = num_train + num_test

all_input_numerical = np.zeros(( total_num , train_input_all.shape[1]))
all_input_numerical[0:num_train,:] = train_input_all
all_input_numerical[num_train:total_num,:] = test_input

all_output = np.zeros((num_train + num_test))
all_output[0:num_train] = train_output_all
all_output[num_train:total_num] = test_output

##########################################################################################
# meta-model 
##########################################################################################
# --> import metamodel -- these metamodels need to have already been created
#		(they can be created with the ROC curve script)
# i.e. enable the whole loop and execute create_ROC_curves.py first 
##########################################################################################
num_train_metamodel_list = [10, 100, 250, 500, 750, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000 ] 
attack_list_all = [] 

for ntm in num_train_metamodel_list:
	# imports GPC metamodel because it performed best -- alternatives will also work 
	num_train_metamodel = ntm
	filename = '../metamodels/pickled_gpc_%itrain.sav'%(num_train_metamodel)
	clf_gpc = pickle.load( open(filename,'rb'))

	# --> run metamodel --> --> -->
	# --> make predictions on the whole test dataset 
	gpc_pred_test = clf_gpc.predict(test_input)

	# --> make predictions on the whole dataset (train + test)
	gpc_pred_all = clf_gpc.predict(all_input_numerical)

	##########################################################################################
	# load / create bitflip list 
	##########################################################################################
	# --> setup with input data 
	all_input_string = [] 
	for kk in range(0,total_num):
		stringy = ''
		for jj in range(0,16):
			if all_input_numerical[kk,jj] == 1:
				stringy += '0'
			else:
				stringy += '1'
		all_input_string.append(stringy)

	all_test_input_string = all_input_string[num_train:]

	# --> search bitflips 
	regenerate_lists = True # this can be kind of slow, so I did it once then pickled
	if regenerate_lists:
		# list of lists that has all of the flip options in it (apologies for ugly code)
		atis_list_of_flips = [] 
		for kk in range(0,num_test):
			orig = all_test_input_string[kk]
			li = [] 
			for jj in range(0,16):
				first_part = orig[0:jj]
				check = orig[jj]
				if check == '0':
					middle_part = '1'
				else:
					middle_part = '0'
				if jj < 15:
					end_part = orig[jj+1:]
				else:
					end_part = ''
		
				stringy = first_part + middle_part + end_part #this was written at the start of the covid-19 pandemic in the USA, therefore to stay sane I had to do silly things like name my strings stringy...deepest apologies for lack of professionalism :)
				li.append(stringy)
			atis_list_of_flips.append(li)
		# --> pickle this
		with open('atis_list_of_flips.pkl', 'wb') as f:
			pickle.dump(atis_list_of_flips, f)
	else: # --> don't re-run, just set regenerate_lists = False
		with open('atis_list_of_flips.pkl', 'rb') as f:
			atis_list_of_flips = pickle.load(f)
	
	if regenerate_lists: #again, this can be slow 
		# for each flip, make an array that says if it's stable or unstable 
		atis_list_of_flips_stable = [] 
		atis_list_of_flips_index = [] 
		for kk in range(0,num_test):
			li = []; li_idx = []
			if test_output[kk] == 0: # it's stable 
				# test if flips are stable 
				for jj in range(0, len(atis_list_of_flips[kk]) ):
					# find in all_test_input_string
					idx = all_input_string.index(atis_list_of_flips[kk][jj])
					li.append(all_output[idx])
					li_idx.append(idx)
				atis_list_of_flips_stable.append(li)
				atis_list_of_flips_index.append(li_idx)
			else:
				atis_list_of_flips_stable.append('starts unstable')
				atis_list_of_flips_index.append('starts unstable')
		# --> pickle this
		with open('atis_list_of_flips_stable.pkl', 'wb') as f:
			pickle.dump(atis_list_of_flips_stable, f)
		with open('atis_list_of_flips_index.pkl','wb') as f:
			pickle.dump(atis_list_of_flips_index, f)
	else: # --> don't re-run, just set regenerate_lists = False
		with open('atis_list_of_flips_stable.pkl', 'rb') as f:
			atis_list_of_flips_stable = pickle.load(f)
		with open('atis_list_of_flips_index.pkl', 'rb') as f:
			atis_list_of_flips_index  = pickle.load(f) 

	##########################################################################################
	ground_truth = atis_list_of_flips_stable
	gt_idx = atis_list_of_flips_index

	# --> create bitflip list for the metamodel 
	if True:
		# --> check stability of meta model 
		mm_compare_to_gt = [] 
		for kk in range(0,num_test):
			li = [] 
			if gpc_pred_test[kk] == 0: # prediction that it's stable
				# test if flips are stable
				if ground_truth[kk] == 'starts unstable':
					mm_compare_to_gt.append('ground truth unstable')
				else:
					for jj in range(0, len(ground_truth[kk])):
						# find index 
						idx = int(gt_idx[kk][jj])
						# look in up in the meta-model output
						li.append(gpc_pred_all[idx])
					mm_compare_to_gt.append(li)
			else:
				mm_compare_to_gt.append('starts unstable')

		with open('mm_compare_to_gt.pkl', 'wb') as f:
			pickle.dump(mm_compare_to_gt , f)
	else:  # --> don't re-run, just set regenerate_lists = False
		with open('mm_compare_to_gt.pkl', 'rb') as f:
			mm_compare_to_gt = pickle.load(f)
	

	######################################################################################
	# setup is above this line ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	######################################################################################
	######################################################################################
	# attack execution is below this line vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
	######################################################################################
	possible_attack_counter = 0 
	NP1_gt_counter = 0; NP2_gt_counter = 0 
	for kk in range(0,num_test):
		if atis_list_of_flips_stable[kk] == 'starts unstable':
			NP1_gt_counter += 1 
			continue
		else:
			possible_attack_counter += np.sum(atis_list_of_flips_stable[kk])
			NP2_gt_counter += 16 - np.sum(atis_list_of_flips_stable[kk])

	print(possible_attack_counter) # 8317 <-- total # of possible attacks in the BIC1 test dataset 

	# counters of all of the different things:
	NP1_counter = 0 # attack not possible 1 -- prediction is unstable and the ground truth is unstable
	NP2_counter = 0 # attack not possible 2 -- prediction is stable/stable, and ground truth is stable/stable
	FA1_counter = 0 # failed attack 1 -- prediction is stable, ground truth is unstable
	FA2_counter = 0 # failed attack 2 -- prediction is stable/unstable, ground truth is stable/stable
	MA1_counter = 0 # missed attack 1 -- prediction is unstable, ground truth is stable
	MA2_counter = 0 # missed attack 2 -- prediction is stable/stable, ground truth is stable/unstable
	SA_counter = 0 # successful attack -- prediction is stable/unstable, ground truth is stable/unstable 

	# compare prediction to round truth via: 
	#	ground_truth list
	#	mm_compare_to_gt list 

	for kk in range(0,num_test):
		if mm_compare_to_gt[kk] == 'starts unstable':
			if ground_truth[kk] == 'starts unstable':
				NP1_counter += 1 
			else:
				MA1_counter += 1 
		else:
			if ground_truth[kk] == 'starts unstable':
				FA1_counter += 1 
			else:
				for jj in range(0,16):
					if mm_compare_to_gt[kk][jj] == 1: # 1 is unstable 
						if ground_truth[kk][jj] == 1:
							SA_counter += 1 
						else:
							FA2_counter += 1 
					else:
						if ground_truth[kk][jj] == 1:
							MA2_counter += 1 
						else:
							NP2_counter += 1 

	attack_list = [SA_counter, NP1_counter, NP2_counter, MA1_counter, MA2_counter, FA1_counter, FA2_counter]
	
	print(ntm)
	print(attack_list)
	attack_list_all.append(attack_list)

##########################################################################################
### --> save info on results and ground truth for plotting
# (plotting script is attack_type_1_plots.py)
##########################################################################################
ground_truth_attack_list = [possible_attack_counter, NP1_gt_counter, NP2_gt_counter] 
np.savetxt('mm_list.txt',np.asarray(num_train_metamodel_list))
np.savetxt('attack1_res_all.txt',np.asarray(attack_list_all))
np.savetxt('attack1_ground_truth.txt',np.asarray(ground_truth_attack_list))







