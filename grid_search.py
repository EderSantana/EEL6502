"""
Projec 1 for Adaptive Filtering Classes - Grid Search for best hyperparameters
"""
import numpy as N
from scipy.io import loadmat
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import lms
import cPickle

data = loadmat('project1.mat')
ref = N.squeeze(data['reference'])
pr  = N.squeeze(data['primary'])

best_w_track = N.zeros_like(ref)
best_out = N.zeros_like(ref)
best_lr = 0.
best_order = 0.

lrates = N.linspace(.0001,.01,5)
orders = N.arange(2,20,3)
SNR = N.zeros((lrates.shape[0], orders.shape[0]))
noSNR = N.zeros_like(SNR)
for i in range(lrates.shape[0]):
    print i
    for j in range(orders.shape[0]):
        f = lms.LMS(pr, ref, orders[j], learning_rate=lrates[i])
        f.train_lms()
        SNR[i,j] = 10*N.log10( N.var(f.error) / N.var( pr*((f.error*pr)**2).sum()/(pr**2).sum() ) )
        #noSNR[i,j] = N.correlate(f.error, pr-pr.mean())
        if SNR[i,j] == N.max(SNR):
            best_w_track = f.w_track.transpose()
            best_out = f.error
            best_lr = lrates[i]
            best_order = orders[j]

cPickle.dump(SNR, open('SNR.pkl','w'), -1)

# Plot SNR Grid
plt.figure()
plt.imshow(SNR)
#plt.xticks(orders)
#plt.yticks(lrates)
plt.ylabel('learning rate $\eta$ = ' + str(lrates))
plt.xlabel('filter order M = ' + str(orders))
plt.savefig('SNR_grid.eps',format='eps')

# Plot Best filter
plt.figure()
plt.plot(best_out)
plt.show()
plt.savefig('best_output_signal.eps',format='eps')

# Plot Best Transfer Function
E = N.abs( N.fft.fft(best_out) )
R = N.abs( N.fft.fft(ref) )
tt = N.linspace(-.5*16000.,.5*16000.,70000)
plt.figure()
plt.plot(tt, E/R)
plt.show()
plt.savefig('best_transfer_func.eps',format='eps')

## Learning curve
plt.figure()
normalizer = N.cumsum(range(1,best_out.shape[0]+1))
MSE = N.cumsum(best_out**2)/normalizer
plt.plot(MSE)
plt.xlim(0,100)
plt.show()
plt.savefig('best_learning_curve.eps',format='eps')

plt.figure()
plt.plot(best_w_track)
plt.show()
plt.savefig('best_weight_track.eps',format='eps')

print "Best learning rate: %f | Best filter order: %i" % (best_lr, best_order)
