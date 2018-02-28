global T_G1, T_other, V_G1, min_G1, L0, EPS,MU,ETA,dt, r_max

#cell-cycle times hours
T_G1, T_other = 2,10
V_G1 = 1 #variance in G1-time
min_G1 = 0.01 #min G1-time
T_D = 17.25663
# T_D = 12.
L0 = 1.0
EPS = 0.05

# Osborne 2017 params
# MU = -50.
# ETA = 1.0
# dt = 0.005 #hours

# Mirams 2012 params
MU = -30.
ETA = 1.
dt = 1./120


r_max = 2.5 #prevents long edges forming in delaunay tri for border tissue

RHO = 1.0
RHO_B = 2**0.5*RHO
GROWTH_RATE = (RHO_B-RHO)/(T_G1+T_other)
DIV_AREA = 3**0.5/2*RHO_B**2