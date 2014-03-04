"""
Projec 1 for Adaptive Filtering Classes - Grid Search for best hyperparameters
"""
import numpy as N
from scipy.io import loadmat
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import lms

data = loadmat('project1.mat')
ref = N.squeeze(data['reference'])
pr  = N.squeeze(data['primary'])

best_w_track = N.zeros_like(ref)
best_out = N.zeros_like(ref)
best_lr = 0.
best_order = 0.

lrates = N.linspace(.0001,.01,10)
orders = N.arange(2,50)
SNR = N.zeros((lrates.shape[0], orders.shape[0]))
for i in lrates:
    print i
    for j in orders:
        f = lms.LMS(pr, ref, j, learning_rate=i)
        f.train_lms()
        SNR[i,j] = -10*N.log10( N.var(filtro.error) / N.var( ref )  )
        if SNR(i,j) == N.max(SNR):
            best_w_track = f.weight_track.transpose()
            best_out = f.error
            best_lr = i
            best_order = j

# Plot SNR Grid
plt.imshow(SNR)
plt.savefig('SNR_gird.eps',format='eps')

# Plot Best filter
plt.figure()
plt.plot(best_out)
plt.show()
plt.savefig('best_output_signal.eps',format='eps')

## Learning curve
plt.figure()
normalizer = N.cumsum(range(1,best_out.shape[0]+1))
MSE = N.cumsum(best_error**2)/normalizer
plt.plot(MSE)
plt.xlim(0,100)
plt.show()
plt.savefig('best_learning_curve.eps',format='eps')

plt.figure()
plt.plot(best_w_track)
plt.show()
plt.savefig('best_weight_track.eps',format='eps')

print "Best learning rate: %f | Best filter order: %i" % (best_lr, best_order)
