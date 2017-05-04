import pysat
import numpy as np
# 
eTensor1=pysat.elasticTensor()
# eTensor.elastic_DB(mtype='ice')
eTensor1.set_radial(vsv=3.57, vsh=3.74, vpv=6.14, vph=6.52, eta=0.87, rho=2790)
eTensor1.rot_dip_strike2(dip=22., strike=37.)
# eTensor.info='Hexagonal Symmetric Media (lower hemisphere)'


kceq1=pysat.Christoffel(etensor=eTensor1)
kceq1.circle(theta=90., dphi=1., group=True)
kceq1.plot_circle(showfig=False)


eTensor2=pysat.elasticTensor()
eTensor2.set_radial(vsv=3.54, vsh=3.71, vpv=6.15, vph=6.47, eta=0.86, rho=2790)
eTensor2.rot_dip_strike2(dip=22., strike=127.)
# eTensor.info='Hexagonal Symmetric Media (lower hemisphere)'

kceq2=pysat.Christoffel(etensor=eTensor2)
kceq2.circle(theta=90., dphi=1., group=True)
kceq2.plot_circle(showfig=False)