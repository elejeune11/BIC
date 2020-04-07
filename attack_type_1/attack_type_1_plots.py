import numpy as np
import matplotlib.pyplot as plt

##########################################################################################
# import data -- must run "execute_attack_type_1.py first!"
##########################################################################################
ground_truth = np.loadtxt('attack1_ground_truth.txt')
mm_tp_list = np.loadtxt('mm_list.txt')
attack_1_res_all = np.loadtxt('attack1_res_all.txt')

##########################################################################################
# percentage performance compared to ground truth  
##########################################################################################

fig = plt.figure()
plt.style.use('el_papers.mplstyle')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig.set_figheight(7)
fig.set_figwidth(7)


SA_percent = [] 
for kk in range(0,attack_1_res_all.shape[0]):
	SA_percent.append(attack_1_res_all[kk,0] / ground_truth[0] * 100 )

NP1_percent = [] 
for kk in range(0,attack_1_res_all.shape[0]):
	NP1_percent.append(attack_1_res_all[kk,1] / ground_truth[1] * 100)

NP2_percent = [] 
for kk in range(0,attack_1_res_all.shape[0]):
	NP2_percent.append(attack_1_res_all[kk,2] / ground_truth[2] * 100)


plt.plot(mm_tp_list,SA_percent,'o-',color=(.75,0,0), linewidth=4.0, markersize=8.0, label='SA')

plt.plot(mm_tp_list,NP1_percent,'s-',color=(.5,.6,.6), linewidth=4.0, markersize=8.0, label='NP-1')

plt.plot(mm_tp_list,NP2_percent,'d-',color=(.75,.85,.85), linewidth=4.0, markersize=8.0, label='NP-2')


plt.ylim((0,100))
plt.legend()
plt.title('Attack 1, Relative Performance')
plt.xlabel('number of training points')
plt.ylabel('\% of ground truth identified')
plt.tight_layout()

plt.savefig('BIC1_attack1_SA_percent')
plt.savefig('BIC1_attack1_SA_percent.eps')

##########################################################################################
# absolute performance w/ all different types of attack 
##########################################################################################

fig = plt.figure()
plt.style.use('el_papers.mplstyle')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig.set_figheight(7)
fig.set_figwidth(7)

# --> plot different absolute numbers (SA, MA1, MA2, FA1, FA2)
plt.plot(mm_tp_list, attack_1_res_all[:,0],'o-',color=(.75,0,0), linewidth=4.0, markersize=8.0, label='SA')
plt.plot(mm_tp_list, attack_1_res_all[:,5],'s-',color=(0,0,.75), linewidth=4.0, markersize=8.0, label='FA-1')
plt.plot(mm_tp_list, attack_1_res_all[:,6],'s-',color=(.5,.5,1.0),  linewidth=4.0, markersize=8.0, label='FA-2')
plt.plot(mm_tp_list, attack_1_res_all[:,3],'d-',color=(.25,.25,.25), linewidth=2.0,label='MA-1')
plt.plot(mm_tp_list, attack_1_res_all[:,4],'d-',color=(.75,.75,.75), linewidth=2.0, label='MA-2')

plt.legend()
plt.title('Attack 1, Absolute Performance')
plt.xlabel('number of training points')
plt.ylabel('absolute number')
plt.tight_layout()

plt.savefig('BIC1_attack1_abs_no')
plt.savefig('BIC1_attack1_abs_no.eps')










