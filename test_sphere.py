import pysat
import numpy as np
# 
eTensor=pysat.elasticTensor()
eTensor.elastic_DB(mtype='ol')

kceq=pysat.Christoffel(etensor=eTensor)
# kceq.sphere(dtheta=1., dphi=1., outfname='sphere001.asdf', group=True)
# kceq.sphere(dtheta=5., dphi=5., outfname='sphere002.asdf', group=True)
kceq.read_asdf(infname='sphere001.asdf')
kceq.plot3d(ptype='rel', stype='rel')
# kceq.set_direction_cartesian(pv=[1.1,2.2,3.3])
# kceq.set_direction_cartesian(pv=[1.1,2.2,3.3])
# kceq.get_phvel()
# kceq.get_grad_mat()
# kceq.get_group_velocity()