import numpy as np 
from scipy import constants
from layer import Layer

'''
TransferMatrix for conductive sheets

			 x
			 ^
			 |
eps1,mu1	 |   eps2, mu2
			 |
_____________|__________________>z
			 |
			 |
             |
             sigma_e, sigma_m


'''

'''
Convension

(hin) = (M)(hout)
'''


import numpy as np
from numpy import linalg as la
import math
from scipy import constants
import copy

eps0 = constants.epsilon_0
mu0 = constants.mu_0
c = constants.c
''' P-polarization

E = (Ex,0,Ez) ; H = (0,H,0)

'''

PEC_CONDUCTIVITY = 1E10 # use this to define PEC conductivity
def M_p(freq, k1z, k2z, eps1, eps2, sigma_e,sigma_m = 0):
	xi  = sigma_e*k2z/(eps2*eps0*freq) # electric cond. term
	zeta = 0#(eps1*eps0*sigma_m/k1z) # magn. cond term
	eta = (eps1/eps2)*(k2z/k1z) # interface term

	M11 = 1+xi+eta+zeta + 0j
	M12 = 1-xi-eta+zeta + 0j
	M21 = 1+xi-eta-zeta + 0j
	M22 = 1-xi+eta-zeta + 0j
	M = 0.5*np.array([[M11,M12],
					   [M21,M22]])
	return M

def M_p_huygens(freq, k1z, k2z, eps1,eps2,sigma_e=0,sigma_m=None):
	'''
	Transfer matrix for p polarization

	Parameters
	----------
	freq : double
	k1z : complex
	k2z : complex
	eps1 : complex
	eps2 : complex
	sigma_e : complex
	sigma_m : complex

	Returns
	-------
	`nd.array`
	'''
	if sigma_m == None: return M_p(freq, k1z, k2z, eps1,eps2,sigma_e)

	eta1 = k1z/(freq*eps1*eps0)
	eta2 = k2z/(freq*eps2*eps0)

	# add a small imaginary part to deal with the zeros
	denom = 2*eta1*(4.0-(sigma_e*sigma_m-1e-12j))

	M11 = (2*eta1+sigma_m)*(eta2*sigma_e+2.)+(2*eta2+sigma_m)*(eta1*sigma_e+2)
	M22 = -(2*eta1-sigma_m)*(eta2*sigma_e-2.)-(2*eta2-sigma_m)*(eta1*sigma_e-2)

	M21 = (2*eta1-sigma_m)*(eta2*sigma_e+2.)+(2*eta2+sigma_m)*(eta1*sigma_e-2)
	M12 = -(2*eta1+sigma_m)*(eta2*sigma_e-2.)-(2*eta2-sigma_m)*(eta1*sigma_e+2)
	M = np.array([[M11,M12],
				[M21,M22]])/denom
	return M

''' S-Polarization

E = (0,E,0) ; H = (Hx,0,Hz)

 '''
def M_s(freq, k1z, k2z, mu1, mu2, sigma_e=0,sigma_m=0):
	xi  = mu1*mu0*freq*sigma_e/k1z
	eta = mu1*k2z/(mu2*k1z)

	zeta = 0
	#print eta,xi

	M11 = 1.0-xi+eta+zeta + 0j
	M12 = 1.0-xi-eta+zeta + 0j
	M21 = 1.0+xi-eta-zeta + 0j
	M22 = 1.0-xi+eta-zeta + 0j
	M = 0.5*np.array([[M11,M12],
					   [M21,M22]])
	return M
def M_s_inv(freq, k1z, k2z, mu1, mu2, sigma,sigma_m=0):
	xi  = mu2*mu0*freq*sigma/k2z
	eta = mu2*k1z/(mu1*k2z)

	zeta = 0#k2z*sigma_m/(freq*mu2*mu0)
	#print eta,xi

	M11 = 1.0-xi+eta+zeta + 0j
	M12 = 1.0-xi-eta-zeta + 0j
	M21 = 1.0+xi-eta+zeta + 0j
	M22 = 1.0+xi+eta-zeta + 0j
	M = 0.5*np.array([[M11,M12],
					   [M21,M22]])
	return M
	
def M_s_huygens(freq, k1z, k2z, mu1,mu2,sigma_e,sigma_m):
	'''
	Transfer matrix for p polarization

	Parameters
	----------
	freq : double
	k1z : complex
	k2z : complex
	eps1 : complex
	eps2 : complex
	sigma_e : complex
	sigma_m : complex

	Returns
	-------
	`nd.array`
	'''
	if sigma_m==None: return M_s(freq, k1z, k2z, mu1,mu2,sigma_e)

	eta1 = freq*mu1*mu0/k1z
	eta2 = freq*mu2*mu0/k2z
	Z0 = np.sqrt(mu0/eps0)

	z21 = eta2/eta1
	s_m = sigma_m/(2.0*Z0)
	s_e = sigma_e*Z0/2.0

	M11 = -1.0/2.0*((z21 + s_m)*(s_e + 1) + (s_m + 1)*(z21*s_e + 1))/(z21*(s_e*s_m - 1))
	M12 = -1.0/2.0*((s_e + 1)*(-s_m + z21) + (s_m + 1)*(s_e*z21 - 1))/(z21*(s_e*s_m - 1))
	M21 = (1.0/2.0)*((s_e - 1)*(s_m + z21) - (s_m - 1)*(s_e*z21 + 1))/(z21*(s_e*s_m - 1))
	M22 = (1.0/2.0)*((s_e - 1)*(-s_m + z21) - (s_m - 1)*(s_e*z21 - 1))/(z21*(s_e*s_m - 1))

	M = np.array([[M11,M12],
				[M21,M22]])

	return M

''' Propagation Matrix '''
def Propagate(k,d):
	a = np.exp(-1j*k*d) + 0j
	b = np.exp(+1j*k*d) + 0j
	P = np.array([[a,0],
				   [0,b]],dtype=np.complex)
	return P

def Propagate_inv(k,d):
	a = np.exp(+1j*k*d) + 0j
	b = np.exp(-1j*k*d) + 0j
	P = np.array([[a,0],
				   [0,b]])
	return P


''' Construct Transfer Matrix for a given list of layers ''' 
class TransferMatrix(object): 

	def __init__(self,structure,freqs=[],angle_of_inc=0.0):
		"""Construct TransferMatrix
		
		Arguments:
			structure {[list]} -- List of layers 
		"""
		self.Ms = np.array([[1.0+0j,0j],[0j,1.0+0j]])
		self.Mp = np.array([[1.0+0j,0j],[0j,1.0+0j]])

		assert isinstance(structure,list) 

		self.structure=structure
		self.angle_inc = 0.0
		self.freqs = []

		self.ts = 0+0j
		self.rs = 0+0j
		self.rp = 0+0j
		self.tp = 0+0j

		self.substrate = None # Set it to Air() 
		self.superstrate = None 


		self.Layers = [self.superstrate] + structure + [self.substrate]

		self._runSetUp = False

		# Check if structure is a list
		if not type(structure) is list: 
			assert "A list of layers is required."
			return -1 

		self.structure = structure

		# Construct the Transfer Matrices
		self.setup()
	
	def setup2(self,angle_inc=0.0):
		'''
		Update the kz values in each layer for a specific angle of incidence
		'''
		ko = self.freqs/constants.c

		# Calculate kx for the superstrate 
		kx = self.superstrate.n*ko*np.sin(angle_inc)
		kz = self.superstrate.n*ko*np.cos(angle_inc)

		self.superstrate.set_kz(kz)

		for li, lay in enumerate(self.structure):
			kl_o = lay.n*ko 
			kl_z = np.sqrt(kl_o**2 - kx**2)
			lay.set_kz(kl_z)
		
		# Substrate
		ksub_o = self.substrate.n*ko
		ksub_z = np.sqrt(ksub_o**2-kx**2)
		self.substrate.set_kz(ksub_z)

		self._runSetUp = True




	def setup(self): 
		# Allocate different memory slots for individual layers 
		# 
		new_structure = []
		for li,lay in enumerate(self.structure): 
			l1 = copy.deepcopy(lay)
			new_structure.append(l1)
		self.structure = new_structure 
		# ---

		# Need an initial layer: layer0
		layer0 = self.structure[0]

		uu = len(self.structure)
		#print("structure layers = ",uu)
		#print("CALCULATING ", len(self.structure))
		self.Ms = np.array([[1,0],[0,1]],dtype=np.complex)
		self.Mp = np.array([[1,0],[0,1]],dtype=np.complex)
		layer0.set_zlim(-10e-3,0.0)

		
		for li,layer1 in enumerate(self.structure[1:len(self.structure)-1]): 
			if isinstance(layer1, Layer): 
				layer1.z_start = layer0.z_end 
				layer1.z_end = layer1.z_start + layer1.thickness
				z_end = layer1.z_start + layer1.d
				
				#print(layer0.name," -> ", layer1.name)
				#layer1.set_zlim(z_start,layer1.)
				interf_s = M_s_huygens(layer1._freq, layer0.kz, layer1.kz, 1, 1, layer1.sigma,sigma_m=layer1.sigma_m)
				
				self.Ms = np.matmul(self.Ms,interf_s)
				#print(interf_s,layer1._freq)

				self.Ms = np.matmul(self.Ms,Propagate(layer1.kz, layer1.thickness))

				#self.Mp *= M_p(layer1._freq, layer0.kz, layer1.kz, layer0.eps, layer1.eps, layer1.sigma,sigma_m=0)
				#self.Mp *= Propagate(layer1.kz, layer1.thickness)
				interf_p = M_p_huygens(layer1._freq, layer0.kz, layer1.kz, layer0.epsilon, layer1.epsilon, layer1.sigma,sigma_m=0)
				self.Mp = np.matmul(self.Mp,interf_p)
				self.Mp = np.matmul(self.Mp, Propagate(layer1.kz, layer1.thickness))
				

				# Next layer
				layer0 = layer1
				
		
		sub = self.structure[-1]
		sub.is_substrate = True
		sub.set_zlim(layer1.z_end,layer1.z_end*4)
		#print(layer0.name," -> ", sub.name)
		
		interf_s = M_s_huygens(layer1._freq, layer0.kz, sub.kz, 1, 1, 0,sigma_m=0)
		self.Ms = np.matmul(self.Ms,interf_s) # --

		interf_p = M_p_huygens(layer1._freq, layer0.kz, sub.kz, layer0.epsilon, sub.epsilon, 0,sigma_m=0)
		#self.Ms = np.matmul(self.Ms,interf_s)
		self.Mp = np.matmul(self.Mp,interf_p) # --
		self._runSetUp = True
 
	def calculate(self): 
		'''
		Calculate transmission and reflection coefficients
		''' 
		if self._runSetUp:
			self.ts = 1.0/self.Ms[0,0] + 0j
			self.rs = self.Ms[1,0]/self.Ms[0,0] + 0j
			self.tp = 1.0/self.Mp[0,0] + 0j
			self.rp = self.Mp[1,0]/self.Mp[0,0] + 0j

	
	def get_M11(self):
		if self._runSetUp:
			return (self.Ms[0,0].real,self.Ms[0,0].imag)
		return None
	
	def get_element(self,i,j):
		if i>2 or i<0 or j>2 or j<0: 
			raise ValueError("Matrix size is 2x2: Elements are [0,0] to [1,1].")
		if self._runSetUp:
			return (self.Ms[i,j].real,self.Ms[i,j].imag)
		return None
	
	def get_field_at_layer(self,polarization):
		'''
		Get the field at each layer
		'''
		if self._runSetUp:
			layer0 = self.structure[-1] # Start f

			uu = len(self.structure)
			# Starting from the superstrate
			
			field_prev = np.array([[1.0],[self.rs]],dtype=complex)
			layer0.field = field_prev
			p = field_prev

			for layer1 in self.structure[1:len(self.structure)-1]: 
				
				interf_s = M_s_huygens(layer1._freq, layer0.kz, layer1.kz, 1, 1, layer1.sigma,sigma_m=0)
				layer1.field = np.matmul(interf_s,p)
				
				
				p =  np.matmul(Propagate(layer1.kz, layer1.thickness),layer1.field)

				layer0 = layer1
				
			#self.structure[-1].field = np.array([[self.ts],[0.0]],dtype=np.complex)







class Layer(object):

	def __call__(self): 
		print(self.name)

	def __init__(self,eps,mu,sigma,d,kz,freq,name="",sigma_m=0j,pec=False,is_substrate=False):
		self.epsilon = eps
		self.n = np.sqrt(eps*mu)
		self.sigma = sigma
		self.sigma_m = sigma_m # magnetic conductivity
		self.mu = mu
		self.name = name
		self.thickness = d
		self.d = d # just for backward compatibility
		self.pec = pec # True to make the layer a PEC
		self._freq = freq

		self.kz = kz
		self.next_layer = None
		self.z_start = -2e-6
		self.z_end = 0.0
		self._angle = 0.0

		self.is_substrate = is_substrate



		if abs(np.real(self.n)) > 0:
			self.wavelength = constants.c/freq/np.real(self.n)
		else:
			self.wavelength = constants.c/freq/100.0

		# Number of cells
		self.Dz = self.d/200.0
		self.Nm = 100#math.ceil(self.thickness/self.Dz)

		self.position = np.linspace(0,d,self.Nm)


		# Field at the interface between this layer and layer i+1
		self.field = np.array([[0j],[0j]],dtype=np.complex)
		# Propagate the field
		self.propagated_field = np.array([[0j],[0j]],dtype=np.complex)

		self.field_array = np.zeros(self.Nm,dtype=np.complex)

	

	def get_field(self,z):
		'''
		get_field() calculates the field at a position z for the layer
		@param float z: distance
		----
		@returns complex
		'''
		
		#uu = np.matmul(Propagate(self.kz,z),self.field)
		uu0 = np.exp(-1j*self.kz*(z-self.z_start))*self.field[0][0]
		uu1 = np.exp(-1j*self.kz*(z-self.z_start))*self.field[1][0]
		
		#if self.is_substrate: 
		#	return self.field[0][0]*np.exp(1j*self.kz*z)
		return uu0+uu1
	
	def set_kz(self,kz):
		'''
		Set normal component of the wavevector for a layer
		'''
		self.kz = kz
	
	def set_zlim(self,z_start,z_end):
		self.z_start = z_start
		self.z_end = z_end
	
	def get_z(self,number=100):
		return np.linspace(self.z_start,self.z_end,number)


class SeminfiniteLayer(Layer):

	def __init__(self,eps,kz,freq,name=""):
		super(SeminfiniteLayer, self).__init__(eps,1,0,0,kz,freq,name=name)

class Layerll(object): 
	'''
	Class for a layer (new June 2021)
	'''
	def __init__(self,epsilon, mu, sigma_e=0,sigma_m=0): 
		self.epsilon = epsilon 
		self.mu = mu 
		self.sigma_e = sigma_e 
		self.sigma_m = sigma_m 

		self.n = np.sqrt(epsilon*mu)
		
''' Field calculation'''
class Field(object):

	def __init__(self,t,r,freq,polarization):
		'''
		Field calculation
		Parameters needed: t, r, structure
		Need M11*t,M21*t
		'''
		self._layers = []
		self._freq = freq

		self.total_thickness = 0.0

		self.transmitted_field = np.array([[t],[0]],dtype=complex)
		self.t = t
		self.r = r
		self.superstrate_field = np.array([[1.0],[r]],dtype=complex)
		self.reflected = np.array([[0],[r]],dtype=complex)
		self.polarization = polarization
		self.layer_name = None

		self.number_of_cells = 0
		self.substrate = None
		self.superstrate_inc  = np.matrix([[1.0],[0]],dtype=complex)
		self.superstrate_r  = None

		self.structure = []

	def __call__(self): 
		for lay in self._layers:
			print(lay.name, " | ")

	def get_layerNames(self):
		self.layer_names = np.array([lay() for lay in self._layers])


	def add_substrate(self,lay):
		self.substrate = lay
		
		self.sub_Nm = lay.Nm
		self.substrate_thickness = lay.d

	def add_superstrate(self,lay):
		self.superstrate = lay # total field
		self.superstrate_r = copy.deepcopy(lay) # reflected field
		self.superstrate_inc = copy.deepcopy(lay) # incident field
		self.superstrate_thickness = lay.d
		self.superstrate_position = np.linspace(-lay.thickness, 0,lay.Nm)


	def add_layer(self,lay):
		'''
		add_layer:
		@param lay: Layer object

		Note: The layers should be added in reverse order.
		'''
		self._layers.append(lay)
		self.total_thickness += lay.thickness
		

	def setup(self):
		'''
		setup()
		Prepare position arrays
		'''
		self.next_layers = np.zeros(len(self._layers)+2,dtype=Layer)
		z_start = 0.0
		z_end = self._layers[0].d
		for ii,lay in enumerate(self._layers):
			z_end = z_start + lay.d
			lay.z_start = z_start
			
			lay.z_end = z_end
			#lay.z_start = z_start
			
			lay.position = np.linspace(lay.z_start,lay.z_end,lay.Nm)
			z_start = lay.z_end
			if ii<len(self._layers)-1:
				lay.next_layer = self._layers[ii+1]


			self.structure.append(np.ones(lay.Nm).flatten().tolist())
		self.structure = np.array(self.structure).flatten()
		self.substrate_position = np.linspace(self.total_thickness, self.total_thickness+self.substrate_thickness,lay.Nm)
		
		


	def calculate_field(self,lay):
		'''
		Calculate field in a layer
		1) Find coefficients from transfer matrix calcs.
		2) E(z) = A*e^(i*kz*(z-zo)) + B*e^(-i*kz*(z-zo))

		At least one layer has to be in self._layers
		'''
		if not isinstance(lay, Layer):
			assert "The given layer does not exist."
			return -1
		if len(self._layers)==0:
			assert "No layers."
			return -1
		
		

		if self.polarization in ['s','p']:
			
			# Find layer in self._layers
			lay_index=np.where(self.get_layerNames() == lay())[0]
		
			if len(lay_index)==0: 
				raise ValueError("Layer is not found.")
				return -1 

			# Calculate transfer matrix
			# up to that layer starting from substrate
			layer1 = self.superstrate
			
			layer2 = self._layers[0]
			lay2_ind  = 0
			tr_matrix = 1.0+0j
			in_layer = False # True if we are in the final layer

			if lay.pec:
				return lambda z: 0

			while not in_layer:
				
				if self.polarization=='s':
					matr = M_s_inv(self._freq, layer1.kz, layer2.kz, 1, 1, layer2.sigma)
					
				else:
					pass

				tr_matrix *= matr
				
				# Propagate only if we are not in the specified layer
				if  lay_index==lay2_ind:
					in_layer=True
					
				else:
					
					tr_matrix *= 1#Propagate_inv(layer2.kz, layer2.thickness)
				lay2_ind = lay2_ind+1
				layer2 = lay
			
					
			f_coeff = tr_matrix[0,0]+tr_matrix[0,1]*self.r
			b_coeff = tr_matrix[1,0]+tr_matrix[1,1]*self.r
			
			
			fu= lambda z: f_coeff*np.exp(+1j*lay.kz*(z-lay.z_start))+b_coeff*np.exp(-1j*lay.kz*(z-lay.z_start))
			return fu(lay.position)
			
		

	def calculate(self):
		'''
		At least one layer has to be in self._layers
		'''
		if len(self._layers)==0:
			assert "No layers."
			return -1
		# Initialize
		self.setup()


		

		if self.polarization in ['s','p']:

			# Propagate in substate layer
			self.substrate.field_array[self.substrate.Nm-1] = self.t
			fl = self.t
			for cell_index in np.arange(self.substrate.Nm-2,-1,-1):
				self.substrate.field_array[cell_index] = fl*np.exp(1j*self.substrate.kz*self.substrate.Dz)
				fl = self.substrate.field_array[cell_index]

			self.substrate.propagated_field = self.transmitted_field
			forward_layer = self.substrate

			for lay in self._layers:
					# M_s(freq, k1z, k2z, mu1, mu2, sigma)
					if self.polarization == 's': 
						tr_matrix = M_s_huygens(self._freq, lay.kz, forward_layer.kz, lay.mu, forward_layer.mu, forward_layer.sigma)
					elif self.polarization == 'p':
						tr_matrix = M_p_huygens(self._freq, lay.kz, forward_layer.kz, lay.epsilon, forward_layer.epsilon, forward_layer.sigma)
					lay.field = np.matmul(tr_matrix,forward_layer.propagated_field)

					fll =  Propagate(lay.kz,lay.d)*lay.field

					print(np.real(np.sum(fll)),np.real(np.sum(lay.field)),self.t)

					# Change previous layer
					forward_layer = lay

					# Fill in the field array
					lay.field_array[0] = np.sum(lay.field)
					field_i = lay.field
					#print lay.field, np.sum(lay.field)
					for cell_index in np.arange(1,lay.Nm):
						field_i = Propagate(lay.kz,lay.Dz)*field_i
						#print field_i
						lay.field_array[cell_index] = np.sum(field_i)
					# Propagate back
					lay.propagated_field = field_i
					#print field_i.shape,field_i
					self.forward_layer = lay

			if self.polarization == 's': 
				tr_matrix = M_s_huygens(self._freq, self.superstrate.kz, forward_layer.kz, self.superstrate.mu, forward_layer.mu, lay.sigma)
			elif self.polarization == 'p':
				tr_matrix = M_p_huygens(self._freq, self.superstrate.kz, forward_layer.kz, self.superstrate.epsilon, forward_layer.epsilon, lay.sigma)
			field_i = np.matmul(tr_matrix,forward_layer.propagated_field)
			#print field_i
			# Propagate in the superstrate layer

			#print self.r, np.real(np.sum(self.superstrate.field))
			#self.superstrate_inc.field_array[self.superstrate_inc.Nm-1] = 1
			flr = self.r #lay.field_array[cell_index]-1.0#self.r
			flinc = 1.0

			self.superstrate.field_array[self.superstrate.Nm-1] = np.sum(field_i) # total field
			self.superstrate_r.field_array[self.superstrate_r.Nm-1] = self.r # reflected field
			self.superstrate_inc.field_array[self.superstrate_inc.Nm-1] = 1.0 # total field
			for cell_index in np.arange(self.superstrate_r.Nm-2,-1,-1):
				self.superstrate_r.field_array[cell_index] = flr*np.exp(+1j*self.superstrate_r.kz*self.superstrate_r.Dz)
				flr = self.superstrate_r.field_array[cell_index]

				self.superstrate_inc.field_array[cell_index] = flinc*np.exp(-1j*self.superstrate_inc.kz*self.superstrate_r.Dz)
				flinc = self.superstrate_inc.field_array[cell_index]

				field_i = np.matmul(Propagate(self.superstrate.kz,self.superstrate.Dz),field_i)
				
				self.superstrate.field_array[cell_index] = np.sum(field_i)
				#flinc = self.superstrate_inc.field_array[cell_index]
