import pysat
import numpy as np
# 
eTensor=pysat.elasticTensor()
# eTensor.elastic_DB(mtype='ol')
# eTensor.set_radial(vsv=3.57, vsh=3.74, vpv=6.14, vph=6.52, eta=0.87, rho=2790)


# eTensor.set_radial(vsv=3.45, vsh=3.61, vpv=6.06, vph=6.24, eta=0.72, rho=2730)
# eTensor.rot_dip_strike2(dip=27, strike=110)


eTensor.set_radial(vsv=3.494, vsh=3.702, vpv=5.94, vph=6.28, eta=0.82, rho=2730)
eTensor.rot_dip_strike2(dip=34, strike=20.)

kceq=pysat.Christoffel(etensor=eTensor)
# kceq.set_direction_cartesian(pv=[1.1,2.2,3.3])
kceq.set_direction_cartesian(pv=[0., 0., 1.])
kceq.get_phvel()
# kceq.get_grad_mat()
# kceq.get_group_velocity()