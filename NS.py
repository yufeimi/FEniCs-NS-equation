from fenics import *
from dolfin import *
from mshr import *
import numpy as np

num_steps = 3400	#time steps
dt = 1e-7	#time step size
T = dt*num_steps #final time
mu = 0		#dynamic viscosity

rho = 1e-3 # density
gamma = 1.4 # specific heat capacity ratio
p0 = 1e6 #the atmospheric pressure
sigmaPMLMax = 10 # Max value for sigma PML
DPML = 1

# Create mesh
channel = Rectangle(Point(0, 0), Point(8, 2))
domain = channel
mesh = generate_mesh(domain, 64)

# Define function spaces
V_e = VectorElement("CG", mesh.ufl_cell(), 1)
Q_e = FiniteElement("CG", mesh.ufl_cell(), 1)

W_e = MixedElement([V_e, Q_e])
W = FunctionSpace(mesh, W_e)

# Define boundaries
inflow = 'near(x[0], 0)'
outflow = 'near(x[0], 8)'
walls = 'near(x[1], 0) || near(x[1], 2)'

# Define PML regions
class PML(SubDomain):
	def inside(self, x, on_boundary):
		tol = 1e-14
		return x[0] >= 7 + tol

PMLvalid = CellFunction('size_t', mesh)
PMLregion = PML()
PMLvalid.set_all(0)
# Mark PML region to be 1 in PMLvalid cell function
PMLregion.mark(PMLvalid,1)

# Define SigmaPML
class SigmaPMLfield(Expression):
	def __init__(self, PMLvalid, sigmaPMLMax, **kwargs):
		self.PMLvalid = PMLvalid
		self.sigmaPMLMax = sigmaPMLMax
	def eval_cell(self, values, x, cell):
		if self.PMLvalid[cell.index] == 0:
			values[0] = 0
			#values[1] = 0
		else:
			values[0] = self.sigmaPMLMax * pow((x[0] - 7)/1, 4)

SigmaPML = SigmaPMLfield(PMLvalid, sigmaPMLMax, degree=0)
# Define trial and test functions
(v, q) = TestFunctions(W)
(qv_t, qp_t) = TestFunctions(W)

# Define functions for solutions at previous and current time steps
w = Function(W)
(u, p) = (as_vector((w[0],w[1])),w[2])
w_PML = Function(W)
(qv, qp) = (as_vector((w_PML[0],w_PML[1])),w_PML[2])
w_n = Function(W)
(u_n, p_n) = (as_vector((w_n[0],w_n[1])),w_n[2])
w_PML_n = Function(W)
(qv_n, qp_n) = (as_vector((w_PML_n[0],w_PML_n[1])),w_PML_n[2])

# Define expressions used in variational forms
U = 0.5*(u_n + u)
n = FacetNormal(mesh)
f = Constant((0, 0)) #body force
k = Constant(dt)
idt = Constant(1./k)
mu = Constant(mu)
rho = Constant(rho)

# Define symmetric gradient
def epsilon(u):
	return sym(nabla_grad(u)) # = nabla_grad(u) + transpose(nabla_grad(u))

# Define stress tensor
def sigma(u, p):
	return 2*mu*epsilon(u) - p*Identity(len(u))

# Define stablized terms in the test functions (Tezduyar 2003)
h = CellSize(mesh)
velocity = u_n
vnorm = sqrt(dot(velocity, velocity))
nu = mu/rho
taum = ( (2.0*idt)**2 + (2.0*vnorm/h)**2 + (4.0*nu/h**2)**2 )**(-0.5)
tauc = taum/rho
taul = h*vnorm/2


# Define the weak form for the momentum equation
F = rho*dot((u - u_n) / k, v)*dx \
+ rho*dot(dot(u, nabla_grad(u)), v)*dx \
+ inner(sigma(u, p), epsilon(v))*dx \
+ dot(p*n, v)*ds - dot(mu*nabla_grad(u)*n, v)*ds \
- dot(f, v)*dx

# Define the modified continuity equation
C = q*((p - p_n) / k)/p0*dx + q*gamma*(p0 + p)*div(u)/p0*dx \
+ q*dot(grad(p), u)/p0*dx


F += C

# assume everthing for the meanflow is zero
# add the PML terms
F += rho*SigmaPML*dot(qv_n,v)*dx + rho*SigmaPML*dot((u_n-qv_n),v)*dx \
+ SigmaPML*dot(qp_n,q)*dx + SigmaPML*dot((p_n-qp_n),q)*dx	

# residual of strong Navier-Stokes
#continuity
r_c = ((p - p_n) / k)/p0 + gamma*(p0 + p)*div(u)/p0 \
+ dot(grad(p), u)/p0 + SigmaPML*qp_n + SigmaPML*(p_n-qp_n)

#momentum
r_m = rho*(idt*(u-u_n) + grad(u)*u) +  \
   - div(sigma(u,p)) + SigmaPML*qv_n + SigmaPML*(u_n-qv_n)

#SUPG and PSPG
F += inner(taul*div(v), r_c)*dx
F += (1/rho)*inner(taum*grad(v)*u*rho + tauc*grad(q), r_m)*dx

# define Jacobian
J = derivative(F, w)

# Weakform for the PML variables
u_n1 = u_n[0]
u_n2 = u_n[1]

F_PML = rho*dot((qv - qv_n) / k, qv_t)*dx \
+ rho*dot(u_n1*u_n.dx(0) , qv_t)*dx \
+ p_n.dx(0)*qv_t[0]*dx \
+ rho*SigmaPML*dot(qv, qv_t)*dx \
+ qp_t*((qp - qp_n) / k)/p0*dx + qp_t*gamma*(p0 + p)*u_n1.dx(0)/p0*dx \
+ qp_t*p_n.dx(0)*u_n1/p0*dx + SigmaPML*qp_t*qp*dx

Jqv = derivative(F_PML, w_PML)

# Create XDMF files for visualization output
xdmffile_u = XDMFFile('results/velocity.xdmf')
xdmffile_p = XDMFFile('results/pressure.xdmf')
xdmffile_qp = XDMFFile('results/qp.xdmf')
xdmffile_qv = XDMFFile('results/qv.xdmf')

# Time-stepping
t = 0

for n in range(num_steps):

	# Update current time
	t += dt

	# Define inflow profile
	inflow_profile = Expression(('6.0*exp(-0.5*pow((t-0.5e-4)/0.15e-4 ,2))','0'),degree=2,t=t)

	# Define boundary conditions
	bcu_inflow = DirichletBC(W.sub(0), inflow_profile, inflow)
	# free to slip walls
	bcu_walls = DirichletBC(W.sub(0).sub(1), Constant(0), walls)
	bcp_outflow = DirichletBC(W.sub(0), Constant((0, 0)), outflow)
	bcs = [bcu_inflow, bcu_walls, bcp_outflow]
	problem = NonlinearVariationalProblem(F, w, bcs, J)
	solver  = NonlinearVariationalSolver(problem)
	solver.parameters['newton_solver']['maximum_iterations'] = 20

	#Define boundary conditions for PML
	bcqv_walls = DirichletBC(W.sub(0).sub(1), Constant(0), walls)
	bcqv_inflow = DirichletBC(W.sub(0), Constant((0,0)), inflow)
	bcsqv = [bcqv_inflow, bcqv_walls]
	problem_pml = NonlinearVariationalProblem(F_PML, w_PML, bcsqv, Jqv)
	solver_pml = NonlinearVariationalSolver(problem_pml)
	solver_pml.parameters['newton_solver']['maximum_iterations'] = 20
	
	# Assemble matrices
	begin("Solving ....")
	print('Time step {}'.format(n))
	solver.solve()
	end()
	begin("Solving PML")
	solver_pml.solve()
	end()
	u, p = w.split()
	qv, qp = w_PML.split()

	# Save solution to file (XDMF/HDF5)
	xdmffile_u.write(u, t)
	xdmffile_p.write(p, t)
	xdmffile_qp.write(qp, t)
	xdmffile_qv.write(qv, t)
	# Save nodal values to file
	# Update previous solution
	w_n.assign(w)

# Hold plot
#interactive()
