from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from mshr import *
import datetime
import sys
import os
##########################################################################################
# --> here we will do a stability test at a single level of applied displacement: 
#		a given input pattern will be reported as either ``stable'' or ``not stable'' 
# The column being simulated has dimensions 16 x 2, and the Young's modulus varies along
#		the height of the column and is dictated by a 16 x 1 input vector 
##########################################################################################
##########################################################################################
# --> input / output setup 
##########################################################################################
idx_label = int(sys.argv[1]) # what index in the input file do you want to simulate 
root_fname = 'results_file_root_name'
fname =  root_fname + str(idx_label) + '_is_stable.txt'
text_file = open(fname, 'w')
text_file.write("CRASH") # this way, if the simulation crashes during either solving 
#		or the eigenvalue analysis we will know, this will be overwritten with the results
text_file.close()

# --> define input pattern 
# data =  np.loadtxt('%i_mat_prop_bitmap.txt'%(idx_label)) # <-- open from saved file 
# --> randomly generate an input pattern and save it (another option)
data = np.zeros((16))
for kk in range(0,16):
	data[kk] = np.random.random()*7.0 + 1 # BIC-3, but can modify 
np.savetxt(str(idx_label) + '_mat_prop_bitmap.txt',data)

##########################################################################################
# --> Problem geometry setup 
##########################################################################################
L = 16 
E = 1
b = 1
h = 2
I = b*h**3.0/12.0
k_factor = 0.5

P_cr = np.pi**2.0 * E * I / (k_factor * L)**2.0
sig_cr = P_cr / (b * h)
eps_cr = sig_cr / E
disp_cr = -1.0 * eps_cr * L  * 1.025

##########################################################################################
# mesh geometry and setup 
##########################################################################################
mref =  320*3 # mesh refinement
quad_flag = 2 # element type: linear vs. quadratic 

##########################################################################################
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
if quad_flag == 1:
	parameters["form_compiler"]["quadrature_degree"] = 1
if quad_flag == 2:
	parameters["form_compiler"]["quadrature_degree"] = 2
##########################################################################################

p_1_x = 0; p_1_y = 0;
p_2_x = h; p_2_y = L; 
rect = Rectangle(Point(p_1_x,p_1_y),Point(p_2_x,p_2_y))
mesh = RectangleMesh(Point(p_1_x,p_1_y),Point(p_2_x,p_2_y),int(mref*(h/L)),int(mref))

##########################################################################################
# build mesh and assign material prop
##########################################################################################
if quad_flag == 1:
	P2 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
if quad_flag == 2:
	P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
TH = P2
W = FunctionSpace(mesh, TH)
V = FunctionSpace(mesh, 'CG', 1)

back = 1.0 # minimum modulus (old version of implementation)
high = 8.0 # maximum modulus (old version of implementation)
nu = 0.3
material_parameters = {'back':back, 'high':high, 'nu':nu}

def make_sum(terms):
	shape, = set(t.ufl_shape for t in terms)
	n = len(terms)
	if n == 0:
		return ufl.zeros(shape)
	if n == 1:
		return terms[0]
	return make_sum(terms[:n//2]) + make_sum(terms[n//2:])

def bitmap(x,y): #there could be a much better way to do this, but this is working within the confines of ufl
	total = 0   
	for jj in range(0, data.shape[0]): # data is the input vector
		const1 = conditional(y>=jj,1,0)
		const2 = conditional(y<jj+1,1,0)
		sum = const1 + const2 
		total = conditional(sum>1, data[jj], total)
	return total 

class GetMat:
	def __init__(self,material_parameters,mesh):
		mp = material_parameters
		self.mesh = mesh
		self.back = mp['back']
		self.high = mp['high']
		self.nu = mp['nu']
	def getFunctionMaterials(self, V):
		self.x = SpatialCoordinate(self.mesh)
		val = bitmap(self.x[0],self.x[1])
		E = val
		effectiveMdata = {'E':E, 'nu':self.nu}
		return effectiveMdata

mat = GetMat(material_parameters, mesh)
EmatData = mat.getFunctionMaterials(V)
E  = EmatData['E']
nu = EmatData['nu']
lmbda, mu = (E*nu/((1.0 + nu )*(1.0-2.0*nu))) , (E/(2*(1+nu)))
matdomain = MeshFunction('size_t',mesh,mesh.topology().dim())
dx = Measure('dx',domain=mesh, subdomain_data=matdomain)

##########################################################################################
# define boundary domains 
##########################################################################################
btm  =  CompiledSubDomain("near(x[1], btmCoord)", btmCoord = p_1_y)
btmBC = DirichletBC(W, Constant((0.0,0.0)), btm)
top  =  CompiledSubDomain("near(x[1], topCoord)", topCoord = p_2_y)
topBCX = DirichletBC(W.sub(0), Constant(0.0),top)

##########################################################################################
# apply traction, and body forces (boundary conditions are within the solver b/c they update)
##########################################################################################
T  = Constant((0.0, 0.0))  # Traction force on the boundary
B  = Constant((0.0, 0.0))

##########################################################################################
# define finite element problem
##########################################################################################
u = Function(W)
du = TrialFunction(W)
v = TestFunction(W)

applied_disp_list = [disp_cr] # could modify to do multiple steps

eigenvalues_all = []  

first_eigenvector_all = []  
second_eigenvector_all = []  

all_stab = [] 

#fname_paraview = File(folder_name + "/paraview.pvd")

for adad in range(0,len(applied_disp_list)):
	applied_disp = applied_disp_list[adad]

	u = Function(W)
	du = TrialFunction(W)
	v = TestFunction(W)

	# Applied compression
	topBCY = DirichletBC(W.sub(1), Constant(applied_disp),top)

	bcs = [btmBC, topBCX, topBCY]

	# Kinematics
	d = len(u)
	I = Identity(d)             # Identity tensor
	F = I + grad(u)             # Deformation gradient
	F = variable(F)

	psi = 1/2*mu*( inner(F,F) - 3 - 2*ln(det(F)) ) + 1/2*lmbda*(1/2*(det(F)**2 - 1) - ln(det(F)))

	f_int = derivative(psi*dx,u,v)
	f_ext = derivative( dot(B, u)*dx('everywhere') + dot(T, u)*ds , u, v)

	# Total potential energy
	Fboth = f_int - f_ext 

	# Derivative of the total potential energy
	dF = derivative(Fboth, u, du)
	
	print('about to solve, time:',datetime.datetime.now())
	
	solve(Fboth == 0, u, bcs, J=dF)

	# Assemble, find eigenvalues/eigenvectors 
	print('about to assemble, time:',datetime.datetime.now())
	A = PETScMatrix()
	assemble(dF, tensor=A)
	for bci in bcs:
		bci.apply(A)

	print('assembly complete, time:',datetime.datetime.now())

	# most effective eigensolver parameters  
	eigensolver = SLEPcEigenSolver(A)
	eigensolver.parameters['spectral_transform'] = 'shift-and-invert'
	eigensolver.parameters['spectral_shift'] = 0.00
	eigensolver.parameters['tolerance'] = 1e-16
	N_solve = 2 # could change to just 1 for this problem
	eigensolver.solve(N_solve)
	
	print('solver done, time:',datetime.datetime.now())

	x_dofs = W.sub(0).dofmap().dofs()
	y_dofs = W.sub(1).dofmap().dofs()
	total_size = len(x_dofs) + len(y_dofs)

	r_list = []; c_list = []; rx_list = []; cx_list = []
	for kk in range(0, N_solve):
		r, c, rx, cx = eigensolver.get_eigenpair(kk)
		r_list.append(r); c_list.append(c);
		rx_list.append(rx); cx_list.append(cx)
		if kk == 0:
			eigenmode = Function(W)
			eigenmode.vector()[:] = rx
			first_eigenvector_all.append(eigenmode)
		if kk == 1:
			eigenmode = Function(W)
			eigenmode.vector()[:] = rx
			second_eigenvector_all.append(eigenmode)

	eigenvalues_all.append(r_list[0])

	if r_list[0] < 0:
		print('UNSTABLE')
		fname =  root_fname + str(idx_label) + '_is_stable.txt'
		text_file = open(fname, 'w')
		text_file.write("UNSTABLE")
		text_file.close()
		all_stab.append(0)
	else:
		print('STABLE')
		fname =  root_fname + str(idx_label) + '_is_stable.txt'
		text_file = open(fname, 'w')
		text_file.write("STABLE")
		text_file.close()
		all_stab.append(1)
		
	#fname_paraview << (u,adad)

















