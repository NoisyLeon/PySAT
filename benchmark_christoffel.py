import pysat
import numpy as np
import christoffel
#

eTensor=pysat.elasticTensor()
eTensor.elastic_DB(mtype='ol')

kceq=pysat.Christoffel(etensor=eTensor)
# kceq.set_direction_cartesian(pv=[1.1,2.2,3.3])
kceq.set_direction_cartesian(pv=[1.1,2.2,3.3])
kceq.get_phvel()
kceq.get_grad_mat()
kceq.get_group_velocity()
kceq.get_enhancement(approx=False)

kceq2=christoffel.Christoffel(stiffness=eTensor.Cvoigt, density=eTensor.rho)
kceq2.set_direction_cartesian(direction=[1.1,2.2,3.3])
# kceq2.get_phase_velocity()
kceq2.get_enhancement(approx=True)