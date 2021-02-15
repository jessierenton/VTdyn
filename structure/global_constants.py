global T_D,T_M,L0,EPS,MU,ETA,dt

L0 = 1.0
A0 = 3**0.5/2.
EPS = 0.05

# params for density dependent model
T_M = None
T_D = 24./0.25
# Osborne 2017 params (scaled)
MU = 6.25
ETA = 1.0
dt = 0.04 #hours


# params for two-player games and pgg (original decoupled model)
# T_M = 1.
# T_D = 12.
# # Osborne 2017 params (original)
# MU = 50
# ETA = 1.0
# dt = 0.005 #hours
