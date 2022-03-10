import numpy as np 
from scipy import constants


class Layer(object):
    """
    Layer class
    """

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