import numpy as np 
from scipy import constants

from transfermatrix import * 


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
