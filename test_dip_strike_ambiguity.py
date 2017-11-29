import pysat
import numpy as np
# 
eTensor=pysat.elasticTensor()
# eTensor.elastic_DB(mtype='ice')
# eTensor.set_radial(vsv=3.494, vsh=3.702, vpv=5.94, vph=6.28, eta=0.82, rho=2730)
# eTensor.rot_dip_strike2(dip=34, strike=20.)

# eTensor.set_radial(vsv=3.45, vsh=3.61, vpv=6.06, vph=6.24, eta=0.72, rho=2730)
# eTensor.rot_dip_strike2(dip=27, strike=110)

eTensor.set_radial(vsv=3.494, vsh=3.702, vpv=5.94, vph=6.28, eta=1.03, rho=2730)
eTensor.rot_dip_strike2(dip=34, strike=20.)

# eTensor.set_radial(vsv=3.45, vsh=3.61, vpv=6.06, vph=6.24, eta=1.03, rho=2730)
# eTensor.rot_dip_strike2(dip=27, strike=110)
eTensor.info='Hexagonal Symmetric Media (lower hemisphere)'


kceq=pysat.Christoffel(etensor=eTensor)
# kceq.sphere(dtheta=1., dphi=1., outfname='group3.asdf', group=True)
# # kceq.sphere(dtheta=5., dphi=5., outfname='sphere002.asdf', group=True)
kceq.read_asdf(infname='group2.asdf')
fig3d= kceq.plot3d(ptype='rel', stype='rel', ds=10)
# kceq.plot2d(ptype='abs', stype='abs', ds=10, theta0=180., hsph='lower', cmap='cv')
# kceq.set_direction_cartesian(pv=[1.1,2.2,3.3])
# kceq.set_direction_cartesian(pv=[1.1,2.2,3.3])
# kceq.get_phvel()
# kceq.get_grad_mat()
# kceq.get_group_velocity()