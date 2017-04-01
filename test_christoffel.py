import pysat
import numpy as np
# 
eTensor=pysat.elasticTensor()
eTensor.elastic_DB(mtype='ol')

kceq=pysat.Christoffel(etensor=eTensor)
# kceq.set_direction_cartesian(pv=[1.1,2.2,3.3])
kceq.set_direction_cartesian(pv=[1.1,2.2,3.3])
kceq.get_phvel()
kceq.get_grad_mat()
kceq.get_group_velocity()