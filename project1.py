"""
Projec 1 for Adaptive Filtering Classes
"""
import numpy as N
from scipy.io import loadmat
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import lms

# Load dataset
data = loadmat('project1.mat')
x = N.squeeze(data['primary'])[0:-2]
x_tau = N.squeeze(data['primary'])[1:-1] 
d = N.squeeze(data['reference'])[2:]

# Visualize 2D error space
_m = 2
range_w = N.linspace(-_m, _m, 120)
w1, w2 = N.meshgrid(range_w, range_w)
J = N.zeros((120,120))
for i in range(J.shape[0]):
    for j in range(J.shape[1]):
        y = range_w[i]*x + range_w[j]*x_tau
        J[i,j] = N.mean(d-y)**2

# Plot error function
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(w1, w2, J, rstride=20, cstride=20, alpha=0.8)
cset = ax.contourf(w1, w2, J, zdir='z', offset=-.1, cmap=cm.coolwarm)
cset = ax.contour(w1, w2, J, zdir='x', offset=-2*_m, cmap=cm.coolwarm)
cset = ax.contour(w1, w2, J, zdir='y', offset=2*_m, cmap=cm.coolwarm)

ax.set_xlabel('w_1')
ax.set_xlim(-2*_m, 2*_m)
ax.set_ylabel('w_2')
ax.set_ylim(-2*_m, 2*_m)
ax.set_zlabel('J')
ax.set_zlim(-0, N.max(J)/2.)

plt.show()
plt.savefig('erro_surface_new.png',format='png')

# Adapt filter
ref = N.squeeze(data['reference'])
pr  = N.squeeze(data['primary'])
filtro = lms.LMS(pr, ref, 3, learning_rate=.01,
         learning_rate_mu=.001)
filtro.train_lms()

plt.figure()
plt.plot(filtro.w_track.transpose())
plt.show()
plt.savefig('weight_track.png',format='png')
plt.savefig('weight_track.eps',format='eps')

plt.figure()
plt.plot(filtro.error)
plt.show()
plt.savefig('output_signal.eps',format='eps')

## Learning curve
plt.figure()
normalizer = N.cumsum(range(1,filtro.error.shape[0]+1))
MSE = N.cumsum(filtro.error**2)/normalizer
plt.plot(MSE)
plt.xlim(0,100)
plt.show()
plt.savefig('learning_curve.eps',format='eps')

SNR = -10*log10( N.var(filtro.error) / N.var( ref )  )
print SNR
