from fenics import *
from dolfin import *
from mshr import *
import numpy as np

num_steps = 4000	#time steps
dt = 1e-7	#time step size
T = dt*num_steps #final time
mu = 1.8e-5		#dynamic viscosity

rho = 1e-3 # density
gamma = 1.4 # specific heat capacity ratio
p0 = 1e6 #the atmospheric pressure

# Create mesh
channel = Rectangle(Point(0, 0), Point(8, 2))
domain = channel
mesh = generate_mesh(domain, 64)

# Define function spaces
V_e = VectorElement("CG", mesh.ufl_cell(), 1)
Q_e = FiniteElement("CG", mesh.ufl_cell(), 1)
W_e = V_e * Q_e
W = FunctionSpace(mesh, W_e)

# Define boundaries
inflow = 'near(x[0], 0)'
outflow = 'near(x[0], 8)'
walls = 'near(x[1], 0) || near(x[1], 2)'


# Define trial and test functions
#(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)


# Define functions for solutions at previous and current time steps
w = Function(W)
(u, p) = (as_vector((w[0],w[1])),w[2])
w_n = Function(W)
(u_n, p_n) = (as_vector((w_n[0],w_n[1])),w_n[2])

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
	return sym(nabla_grad(u))

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
+ inner(sigma(U, p_n), epsilon(v))*dx \
+ dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds \
- dot(f, v)*dx

# Define the modified continuity equation
C = q*((p - p_n) / k)/p0*dx + q*gamma*(p0 + p)*div(u)/p0*dx \
+ q*dot(grad(p), u)/p0*dx


F += C

# residual of strong Navier-Stokes
#continuity
r_c = ((p - p_n) / k)/p0 + gamma*(p0 + p)*div(u)/p0 \
+ dot(grad(p), u)/p0

#momentum
r_m = rho*(idt*(u-u_n) + grad(u)*u) +  \
   - div(sigma(u,p))

#SUPG and PSPG
F += inner(taul*div(v), r_c)*dx
F += (1/rho)*inner(taum*grad(v)*u*rho + tauc*grad(q), r_m)*dx

# define Jacobian
J = derivative(F, w)

a = lhs(F)
L = rhs(F)

# Create XDMF files for visualization output
xdmffile_u = XDMFFile('results/velocity.xdmf')
xdmffile_p = XDMFFile('results/pressure.xdmf')

# Time-stepping
t = 0

for n in range(num_steps):

	# Update current time
	t += dt

	# Define inflow profile
	inflow_profile = Expression(('6.0*exp(-0.5*pow((t-0.5e-4)/0.15e-4 ,2))','0'),degree=2,t=t)

	# Define the outflow total pressure
	
	#outflow_profile = Expression(('-0.5*rho*pow(u,2)'), rho=rho, u=w_n[0])

	# Define boundary conditions
	bcu_inflow = DirichletBC(W.sub(0), inflow_profile, inflow)
	# free to slip walls
	bcu_walls = DirichletBC(W.sub(0).sub(1), Constant(0), walls)
	#bcp_outflow = DirichletBC(W.sub(1), outflow_profile, outflow)
	bcs = [bcu_inflow, bcu_walls]
	problem = NonlinearVariationalProblem(F, w, bcs, J)
	solver  = NonlinearVariationalSolver(problem)
	solver.parameters['newton_solver']['maximum_iterations'] = 20
	
	# Assemble matrices
	begin("Solving ....")
	print('Time step {}'.format(n))
	solver.solve()
	end()
	u, p = w.split()

	# Save solution to file (XDMF/HDF5)
	xdmffile_u.write(u, t)
	xdmffile_p.write(p, t)
	# Save nodal values to file
	# Update previous solution
	w_n.assign(w)

# Hold plot
#interactive()
