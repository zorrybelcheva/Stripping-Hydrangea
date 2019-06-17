import numpy as np
import matplotlib.pyplot as plt


def gaussian(x, mu, sigma):
    return np.exp(-(x-mu)**2/2/sigma**2)/np.sqrt(2*np.pi*sigma*sigma)


def plot(filename, particles, limits, plotname):
    file = np.loadtxt(filename)
    dx = file[:, 0]*1000000
    dy = file[:, 1]*1000000
    dz = file[:, 2]*1000000
    dist = file[:, 3]*1000000

    meanx = np.average(dx)
    meany = np.average(dy)
    meanz = np.average(dz)
    meand = np.average(dist)
    x = np.linspace(limits[0], limits[1], 2000)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 7))
    ax1.hist(dist, bins=50, facecolor='red', alpha=0.5, label='distance (abs)', density=True)
    ax1.axvline(meand, c='maroon', label='mean(dist) = {:1.0f} pc'.format(meand))
    ax1.legend(loc='upper right', fontsize='x-small')
    ax1.set_xlim(limits)

    ax2.hist(dx, bins=50, facecolor='royalblue', alpha=0.5, label='dx', density=True)
    ax2.axvline(meanx, c='royalblue', label='mean(dx) = {:1.0f} pc'.format(meanx))
    ax2.plot(x, gaussian(x, meanx, np.std(dx)), c='k', linestyle='--', linewidth=1,
             label='G($\mu$={:1.0f}, $\sigma$={:1.0f})'.format(meanx, np.std(dx)))
    ax2.legend(loc='upper right', fontsize='x-small')
    ax2.set_xlim(limits)

    ax3.hist(dy, bins=50, facecolor='orange', alpha=0.5, label='dy', density=True)
    ax3.axvline(meany, c='orange', label='mean(dy) = {:1.0f} pc'.format(meany))
    ax3.plot(x, gaussian(x, meany, np.std(dy)), c='k', linestyle='--', linewidth=1,
             label='G($\mu$={:1.0f}, $\sigma$={:1.0f})'.format(meany, np.std(dy)))
    ax3.legend(loc='upper right', fontsize='x-small')
    ax3.set_xlim(limits)

    ax4.hist(dz, bins=50, facecolor='green', alpha=0.5, label='dz', density=True)
    ax4.axvline(meanz, c='green', label='mean(dz) = {:1.0f} pc'.format(meanz))
    ax4.plot(x, gaussian(x, meanz, np.std(dz)), c='k', linestyle='--', linewidth=1,
             label='G($\mu$={:1.0f}, $\sigma$={:1.0f})'.format(meanz, np.std(dz)))
    ax4.legend(loc='upper right', fontsize='x-small')
    ax4.set_xlim(limits)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.title(particles+' Particles\nDistribution of (bootstrapped) displacements from CoM, galID = 2373')
    plt.xlabel('Displacement from CoM, pc', labelpad=12)
    plt.ylabel('Normalised bin count', labelpad=20)
    fig.tight_layout()

    plt.savefig(plotname, dpi=300)


limits_dm = (-250, 450)
limits_sm = (-60, 90)

plot(filename='displacement_stars.txt', particles='Stellar', limits=limits_sm,
     plotname='/home/belcheva/Desktop/Project/displacements_stars.png')
