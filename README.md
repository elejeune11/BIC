# Buckling Instability Classification (BIC) dataset 
### Link to permanent home of the dataset: https://open.bu.edu/handle/2144/40085
### Link to manuscript: (forthcoming)

## FEA code to generate BIC 
The Buckling Instability Classification (BIC) datasets contain the results of finite element simulations where a heterogeneous column is subject to a fixed level of applied displacement and is classified as either Stable or Unstable. Each model input is a 16x1 vector where the entries of the vector dictate the Young's Modulus (E) of the corresponding portion of the physical column domain. Each input file has 16 columns one for each vector entry. For each 16x1 vector input, there is a single output that indicates if the column was stable or unstable at the fixed level of applied displacement. An output value of 0 indicates stable, and an output value of 1 indicates unstable. In BIC-1, we only allow two possible discrete values for E: E=1 or E=4. In BIC-2, we allow three possible discrete values for E: E=1, E=4, or E=7. In BIC-3, we allow continuous values (to three digits of precision) of E in the range E=1–8. BIC-1 consists of 65,536 simulation results. This exhausts the entire possible input domain. BIC-2 consists of 100,000 simulation results. This is less than 1% of the entire possible input domain. BIC-3 also consists of 100,000 simulation results. This is a tiny fraction of the entire possible input domain. 

In the folder generate_dataset is the script run_single_simulation.py. To run this script, FEniCS (https://fenicsproject.org/) installed with PETSc and SLEPc is required. At the start of the script, it is necessary to specify either the input file directory or an algorithm for generating input vectors. Each run of the script will save a textfile that says if the column with material properties dictated by the input vector was stable or unstable at the given level of applied displacement. 


## Code to create metamodels
In the folder metamodels is the script create_ROC_curves.py (and matplotlib style file el_papers.mplstyle). This script is used to creat the ROC curves in the manuscript associated with the BIC dataset (link forthcoming). First, metamodels (support vector machine, neural network, Gaussian process classifier) are created from training data. Next, the performance on test data is evaluated. This simple script uses scikit-learn (https://scikit-learn.org/stable/supervised_learning.html#supervised-learning) to create the metamodels. 


## Code to execute a type 1 adversarial attack
In the folder attack_type_1 is the script execute_attack_type_1.py. This script is used to execute the type 1 attack in the manuscript associated with the BIC dataset (link forthcoming). The script attack_type_1_plots.py is used to generate the plots in the results section of the manuscript (link forthcoming). 


## Code to execute a type 2 adversarial attack 
In the folder attack_type_2 is the script execute_attack_type_2.py. This script is used to execute the type 2 attack in the manuscript associated with the BIC dataset (link forthcoming). The script attack_type_2_plots.py is used to generate the plots in the results section of the manuscript (link forthcoming). 





