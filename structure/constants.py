#cell-cycle times hours
T_G1, T_other = 2,10
V_G1 = 1 #variance in G1-time
min_G1 = 0.01 #min G1-time

L0 = 1.0
EPS = 0.05


MU = -50.
ETA = 1.0
dt = 0.005 #hours
r_max = 2.5 #prevents long edges forming in delaunay tri for border tissue