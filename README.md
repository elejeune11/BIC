# Buckling Instability Classification (BIC) dataset 
### Link to permanent home of the dataset: (forthcoming)
### Link to manuscript: (forthcoming)

## FEA code to generate BIC 
The Buckling Instability Classification (BIC) datasets contain the results of finite element simulations where a heterogeneous column is subject to a fixed level of applied displacement and is classified as either Stable or Unstable. Each model input is a 16x1 vector where the entries of the vector dictate the Young's Modulus (E) of the corresponding portion of the physical column domain. Each input file has 16 columns one for each vector entry. For each 16x1 vector input, there is a single output that indicates if the column was stable or unstable at the fixed level of applied displacement. An output value of 0 indicates stable, and an output value of 1 indicates unstable. In BIC-1, we only allow two possible discrete values for E: E=1 or E=4. In BIC-2, we allow three possible discrete values for E: E=1, E=4, or E=7. In BIC-3, we allow continuous values (to three digits of precision) of E in the range E=1–8. BIC-1 consists of 65,536 simulation results. This exhausts the entire possible input domain. BIC-2 consists of 100,000 simulation results. This is less than 1% of the entire possible input domain. BIC-3 also consists of 100,000 simulation results. This is a tiny fraction of the entire possible input domain. 

In the folder generate_dataset is the script run_single_simulation.py. To run this script, FEniCS (https://fenicsproject.org/) installed with PETSc and SLEPc is required. At the start of the script, it is necessary to specify either the input file directory or an algorithm for generating input vectors. Each run of the script will save a textfile that says if the column with material properties dictated by the input vector was stable or unstable at the given level of applied displacement. 


## Code to create metamodels



## Code to execute a type 1 adversarial attack



## Code to execute a type 2 adversarial attack 





