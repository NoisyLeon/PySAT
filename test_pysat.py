import pysat
import numpy as np
# 
eTensor=pysat.elasticTensor()
eTensor.elastic_DB(mtype='ice')

R=pysat.euler2mat(1,2,3)
# eTensor.Voigt2Cijkl()
# eTensor.set_radial(8.7, 4.5, 3.3, 0.1, 0.5, 1)
# eTensor.set_love(A=13.961, C=15.013, L=3.21, N=3.404, F=5.765)
# eTensor.set_thomsen(vp=5.3, vs=3.2, eps=0.1, gamma=0.05, delta=0.00, rho=4000)
# # eTensor.Voigt2Cijkl()
# eTensor.rotB([0,1,0], 90.)
# eTensor.rotB([1,0,0], 90.)
# eTensor.rotTB([0,0,1], 90.)
# eTensor.Cijkl2Voigt()
# 
# # 
# eTensor2=pysat.elasticTensor()
# # # eTensor.Voigt2Cijkl()
# # 
# eTensor2.set_love(A=13.961, C=15.013, L=3.21, N=3.404, F=5.765)
# # eTensor.Voigt2Cijkl()
# eTensor2.rotT([0,1,0], 90.)

# eTensor2.rotB([1,0,0], 90.)
# eTensor2.rotB([0,0,1], 90.)