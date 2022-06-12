import numpy as np
import skfuzzy as sk
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def peat_calc(vel_indx, dist_indx):
    suma, area = 0, 0

    R = [[min(vel_auto[vel_indx], dist_auto[dist_indx]) for vel_auto in vel_qual] for dist_auto in dist_qual]
    
    vel_peat = [max(R[3][0], R[3][1]),
                max(R[2][0],R[2][1]),
                max(R[1][0], R[1][1], R[3][2]),
                max(R[0][0],R[0][1], R[2][2]),
                max(R[0][2], R[1][2])]
    
    output = np.zeros(graph3.shape)

    for i,vel in enumerate(graph3):
        output[i] = max([min(vel_peat[index], peat_qual[index][i]) for index in range(len(vel_peat))])
        suma += vel * output[i]
        area += output[i]

    return suma/area

#Defining the Numpy array for Tip Quality
graph1 = np.arange(5, 90) #graph vel auto
graph2 = np.arange(5, 20) #graph dist auto
graph3 = np.arange(0, 10, 0.1) #graph peat vel

x = int(input('Pon la velocidad del vehículo (0 - 85):\n'))
y = int(input('Pon la distancia del vehículo (0 - 15):\n'))

#Defining the Numpy array for Triangular membership functions
vel_lo = sk.trapmf(graph1, [5, 5, 15 , 25])
vel_me = sk.trimf(graph1, [10, 45, 80])
vel_hi = sk.trapmf(graph1, [70, 75, 90, 90])
vel_qual = [vel_lo, vel_me, vel_hi]

dist_lo = sk.trapmf(graph2, [5, 5, 6 , 7])
dist_me = sk.trapmf(graph2, [6, 8, 10, 12])
dist_midh = sk.trimf(graph2, [10, 13, 16])
dist_hi = sk.trapmf(graph2, [15, 16, 20, 20])
dist_qual = [dist_lo, dist_me, dist_midh, dist_hi]

peat_lo = sk.gbellmf(graph3, 1,2,0)
peat_midl = sk.gbellmf(graph3, 1,3,3)
peat_me = sk.gbellmf(graph3, 1,2,5)
peat_midh = sk.gbellmf(graph3, 1,3,7)
peat_hi = sk.gbellmf(graph3, 1,2,10)
peat_qual = [peat_lo, peat_midl, peat_me, peat_midh, peat_hi]

R1 = min(vel_lo[x], dist_lo[y])
R2 = min(vel_lo[x], dist_me[y])
R3 = min(vel_lo[x], dist_midh[y])
R4 = min(vel_lo[x], dist_hi[y])

R5 = min(vel_me[x], dist_lo[y])
R6 = min(vel_me[x], dist_me[y])
R7 = min(vel_me[x], dist_midh[y])
R8 = min(vel_me[x], dist_hi[y])

R9 = min(vel_hi[x], dist_lo[y])
R10 = min(vel_hi[x], dist_me[y])
R11 = min(vel_hi[x], dist_midh[y])
R12 = min(vel_hi[x], dist_hi[y])

p_ml = max(R4, R8)
p_l = max(R3, R7)
p_m = max(R2, R6, R12)
p_a = max(R1, R5, R11)
p_ma = max(R9, R10)
array_peaton = [p_ml, p_l, p_m, p_a, p_ma]

z1 = np.fmin(p_ml, peat_lo)
z2 = np.fmin(p_l, peat_midl)
z3 = np.fmin(p_m, peat_me)
z4 = np.fmin(p_a, peat_midh)
z5 = np.fmin(p_ma, peat_hi)

aggregated = np.fmax(z1, np.fmax(z2, np.fmax(z3, np.fmax(z4, z5))))
 
plt.subplot(2, 2, 1)
plt.plot(graph1,vel_lo)
plt.plot(graph1,vel_me)
plt.plot(graph1,vel_hi)
plt.ylabel('\u03BC')
plt.xlabel('km/h')
plt.legend(['lento', 'medio', 'rapido'])
plt.title("Velocidad del coche")

plt.subplot(2, 2, 2)
plt.plot(graph2,dist_lo)
plt.plot(graph2,dist_me)
plt.plot(graph2,dist_midh)
plt.plot(graph2,dist_hi)
plt.ylabel('\u03BC')
plt.xlabel('m')
plt.legend(['muy cerca', 'algo cerca', 'algo lejos', 'muy lejos'])
plt.title("Distancia del coche")

plt.subplot(2, 2, 3)
plt.plot(graph3,peat_lo)
plt.plot(graph3,peat_midl)
plt.plot(graph3,peat_me)
plt.plot(graph3,peat_midh)
plt.plot(graph3,peat_hi)
plt.ylabel('\u03BC')
plt.xlabel('km/h')
plt.legend(['muy lento', 'lento', 'promedio', 'rapido', 'muy rapido'])
plt.title("Velocidad peatonal")

plt.subplot(2, 2, 4)
plt.plot(graph3, aggregated)
plt.ylim([0.0, 1.0])
plt.ylabel('\u03BC')
plt.xlabel('m/s')
plt.legend(['resultado'])
plt.title("Velocidad peatonal final")
plt.show()

plt.figure(1)
ax = plt.axes(projection = '3d')
x_3d, y_3d = np.meshgrid(graph2, graph1)
z_3d = np.zeros((len(graph1), len(graph2)))
for i in range(len(graph2)):
    for j in range(len(graph1)):
        z_3d[j, i] = peat_calc(np.round((graph1[j]-5)), np.round((graph2[i]-5)))

plt.figure(1)
ax.plot_surface(x_3d, y_3d, z_3d, rstride = 1, cstride = 1, cmap = 'inferno', edgecolor = 'none')
ax.set_xlabel("Distancia del auto [m] X")
ax.set_ylabel("Velocidad del auto [km/h] Y")
ax.set_zlabel("Velocidad del peaton [km/h] Z")
ax.set_title("Superficie de Control")
plt.show()