import pysa
import numpy as np
# 
eTensor=pysa.elasticTensor()
# eTensor.Voigt2Cijkl()
eTensor.set_radial(8.7, 4.5, 3.3, 0.1, 0.5, 1)
# eTensor.set_love(A=13.961, C=15.013, L=3.21, N=3.404, F=5.765)
# # eTensor.Voigt2Cijkl()
# eTensor.rotB([0,1,0], 90.)
# # eTensor.Cijkl2Voigt()
# 
# 
# eTensor2=pysa.elasticTensor()
# # eTensor.Voigt2Cijkl()
# 
# eTensor2.set_love(A=13.961, C=15.013, L=3.21, N=3.404, F=5.765)
# # eTensor.Voigt2Cijkl()
# eTensor2.rotT([0,1,0], 90.)