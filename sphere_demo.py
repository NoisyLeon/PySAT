import pysat
import numpy as np
# 
eTensor=pysat.elasticTensor()
eTensor.elastic_DB(mtype='ol')

kceq=pysat.Christoffel(etensor=eTensor)
kceq.sphere(dtheta=1., dphi=1., outfname='sphere001.asdf', group=True)
kceq.read_asdf(infname='sphere001.asdf')
kceq.plot3d(ptype='rel', stype='rel')
kceq.plot2d(ptype='abs', stype='abs', ds=10)
