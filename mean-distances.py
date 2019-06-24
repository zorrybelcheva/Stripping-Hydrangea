import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'][0] = 'palatino'


def read_mean_std(galID):
    outputdir = '/home/belcheva/PycharmProjects/project/Stripping-Hydrangea/output/'

    # Dark Matter particles:
    filename = outputdir+'displacement_1_'+str(galID)+'.txt'
    file = np.loadtxt(filename)
    dx = file[:, 0] * 1000000
    dy = file[:, 1] * 1000000
    dz = file[:, 2] * 1000000
    dist = file[:, 3] * 1000000

    meanxdm = np.average(dx)
    meanydm = np.average(dy)
    meanzdm = np.average(dz)
    meanddm = np.average(dist)

    stdxdm = np.std(dx)
    stdydm = np.std(dy)
    stdzdm = np.std(dz)
    stdddm = np.std(dist)

    # Stellar Matter particles:
    filename = outputdir+'displacement_4_'+str(galID)+'.txt'
    file = np.loadtxt(filename)
    dx = file[:, 0] * 1000000
    dy = file[:, 1] * 1000000
    dz = file[:, 2] * 1000000
    dist = file[:, 3] * 1000000

    meanxsm = np.average(dx)
    meanysm = np.average(dy)
    meanzsm = np.average(dz)
    meandsm = np.average(dist)

    stdxsm = np.std(dx)
    stdysm = np.std(dy)
    stdzsm = np.std(dz)
    stddsm = np.std(dist)

    mux = meanxdm - meanxsm
    muy = meanydm - meanysm
    muz = meanzdm - meanzsm

    # mud = np.abs(meanddm - meandsm)
    mud = np.sqrt(mux*mux+muy*muy+muz*muz)

    stdx = np.sqrt(stdxdm**2+stdxsm**2)
    stdy = np.sqrt(stdydm**2+stdysm**2)
    stdz = np.sqrt(stdzdm**2+stdzsm**2)
    # stdd = np.sqrt(stdddm**2+stddsm**2)

    stdd = np.sqrt((mux/mud)**2*stdx + (muy/mud)**2*stdy + (muz/mud)**2*stdz)

    mu = [mux, muy, muz, mud]
    std = [stdx, stdy, stdz, stdd]

    return mu, std


galIDs = np.loadtxt('output/lessDM.txt', unpack=True)[0].astype(int)
mdm = np.loadtxt('output/lessDM.txt', unpack=True)[1]
msm = np.loadtxt('output/lessDM.txt', unpack=True)[2]

mx, my, mz, md = [], [], [], []
sx, sy, sz, sd = [], [], [], []

for galID in galIDs:
    mu, std = read_mean_std(galID)
    mx.append(mu[0])
    my.append(mu[1])
    mz.append(mu[2])
    md.append(mu[3])

    sx.append(std[0])
    sy.append(std[1])
    sz.append(std[2])
    sd.append(std[3])
    # print(mu, std)

size = len(mx)
print(size)

i = np.where(galIDs == 2373)[0]
md = np.array(md)

plt.figure()
plt.errorbar(mdm[:size], mx, yerr=sx, fmt='.', c='royalblue', label='x')
plt.errorbar(mdm[:size], my, yerr=sy, fmt='.', c='orange', label='y')
plt.errorbar(mdm[:size], mz, yerr=sz, fmt='.', c='green', label='z')
plt.legend(fontsize='small')
plt.xlabel('Dark matter mass, $\log{(M/M_{\odot})}$')
plt.ylabel('Coordinate displacement, [pc]')

plt.figure()
plt.errorbar(mdm[:size], md, yerr=sd, fmt='.', c='k')
# plt.scatter(mdm[i], md[i], marker='x', s=2, c='red')
plt.xlabel('Dark matter mass, $\log{(M/M_{\odot})}$')
plt.ylabel('Displacement in distance, [pc]')

plt.figure()
plt.errorbar(msm[:size], mx, yerr=sx, fmt='.', c='royalblue', label='x')
plt.errorbar(msm[:size], my, yerr=sy, fmt='.', c='orange', label='y')
plt.errorbar(msm[:size], mz, yerr=sz, fmt='.', c='green', label='z')
plt.legend(fontsize='small')
plt.xlabel('Stellar matter mass, $\log{(M/M_{\odot})}$')
plt.ylabel('Coordinate displacement, [pc]')

plt.figure()
plt.errorbar(msm[:size], md, yerr=sd, fmt='.', c='k')
plt.xlabel('Stellar matter mass, $\log{(M/M_{\odot})}$')
plt.ylabel('Displacement in distance, [pc]')
