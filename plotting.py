import numpy as np
import matplotlib.pyplot as plt


def gaussian(x, mu, sigma):
    return np.exp(-(x-mu)**2/2/sigma**2)/np.sqrt(2*np.pi*sigma*sigma)


def plot(filename, particles, limits, plotname, combined=False):
    if not combined:
        file_dm = np.loadtxt(filename)
        dx = file_dm[:, 0]*1000000
        dy = file_dm[:, 1]*1000000
        dz = file_dm[:, 2]*1000000
        dist = file_dm[:, 3]*1000000

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
        plt.title(particles+' Particles\nDistribution of (bootstrapped) displacements from CoM, galID=' +
                  filename.split('.')[0].split('_')[-1])
        plt.xlabel('Displacement from CoM, pc', labelpad=12)
        plt.ylabel('Normalised bin count', labelpad=20)
        fig.tight_layout()

        plt.savefig(plotname, dpi=300)

    else:
        pass
        # file_dm = np.loadtxt(filename)
        # dx = file_dm[:, 0] * 1000000
        # dy = file_dm[:, 1] * 1000000
        # dz = file_dm[:, 2] * 1000000
        # dist = file_dm[:, 3] * 1000000
        #
        # file_sm = np.loadtxt(filename.split('_')[0]+'_4_'+filename.split('_')[2])
        # sx = file_sm[:, 0] * 1000000
        # sy = file_sm[:, 1] * 1000000
        # sz = file_sm[:, 2] * 1000000
        # s_dist = file_sm[:, 3] * 1000000
        #
        # meanx = np.average(sx)
        # meany = np.average(sy)
        # meanz = np.average(sz)
        # meand = np.average(s_dist)
        #
        # Min = int(min(min(dx), min(dy), min(dz), min(dist), min(sx), min(sy), min(sz), min(s_dist)).round())
        # Max = int(min(max(dx), max(dy), max(dz), max(dist), max(sx), max(sy), max(sz), max(s_dist)).round())
        #
        # counts_x_dm, bins_x_dm = np.histogram(dx, bins=range(Min, Max, 10))
        # counts_y_dm, bins_y_dm = np.histogram(dy, bins=range(Min, Max, 10))
        # counts_z_dm, bins_z_dm = np.histogram(dz, bins=range(Min, Max, 10))
        # counts_x_sm, bins_x_sm = np.histogram(sx, bins=range(Min, Max, 10))
        # counts_y_sm, bins_y_sm = np.histogram(sy, bins=range(Min, Max, 10))
        # counts_z_sm, bins_z_sm = np.histogram(sz, bins=range(Min, Max, 10))
        #
        # counts_x_dm -= counts_x_sm
        # counts_y_dm -= counts_y_sm
        # counts_z_dm -= counts_z_sm
        #
        # x = np.linspace(limits[0], limits[1], 2000)
        #
        # fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 7))
        # ax1.hist(dist, bins=50, facecolor='red', alpha=0.5, label='distance (abs)', density=True)
        # ax1.axvline(meand, c='maroon', label='mean(dist) = {:1.0f} pc'.format(meand))
        # ax1.legend(loc='upper right', fontsize='x-small')
        # ax1.set_xlim(limits)
        #
        # # ax2.hist(dx, bins=50, facecolor='royalblue', alpha=0.5, label='dx', density=True)
        # ax2.bar(bins_x_dm[:-1]+5, counts_x_dm, width=10, facecolor='royalblue', alpha=0.5, label='dx')
        # ax2.axvline(meanx, c='royalblue', label='mean(dx) = {:1.0f} pc'.format(meanx))
        # ax2.plot(x, gaussian(x, np.average(dx) - meanx, np.sqrt(np.std(dx)+np.std(sx))), c='k', linestyle='--', linewidth=1,
        #          label='G($\mu$={:1.0f}, $\sigma$={:1.0f})'.format(meanx, np.std(dx)))
        # ax2.legend(loc='upper right', fontsize='x-small')
        # ax2.set_xlim(limits)
        #
        # ax3.hist(dy, bins=50, facecolor='orange', alpha=0.5, label='dy', density=True)
        # ax3.axvline(meany, c='orange', label='mean(dy) = {:1.0f} pc'.format(meany))
        # ax3.plot(x, gaussian(x, meany, np.std(dy)), c='k', linestyle='--', linewidth=1,
        #          label='G($\mu$={:1.0f}, $\sigma$={:1.0f})'.format(meany, np.std(dy)))
        # ax3.legend(loc='upper right', fontsize='x-small')
        # ax3.set_xlim(limits)
        #
        # ax4.hist(dz, bins=50, facecolor='green', alpha=0.5, label='dz', density=True)
        # ax4.axvline(meanz, c='green', label='mean(dz) = {:1.0f} pc'.format(meanz))
        # ax4.plot(x, gaussian(x, meanz, np.std(dz)), c='k', linestyle='--', linewidth=1,
        #          label='G($\mu$={:1.0f}, $\sigma$={:1.0f})'.format(meanz, np.std(dz)))
        # ax4.legend(loc='upper right', fontsize='x-small')
        # ax4.set_xlim(limits)
        #
        # fig.add_subplot(111, frameon=False)
        # plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        # plt.title(particles + ' Particles\nDistribution of (bootstrapped) displacements from CoM, galID = 2373')
        # plt.xlabel('Displacement from CoM, pc', labelpad=12)
        # plt.ylabel('Normalised bin count', labelpad=20)
        # fig.tight_layout()

        # plt.savefig(plotname, dpi=300)


limits_dm = (-150, 150)
limits_sm = (-60, 90)

plotdir = '/home/belcheva/PycharmProjects/project/Stripping-Hydrangea/plots/'

plot(filename='output/displacement_1_2373.txt', particles='Dark Matter', limits=limits_dm,
     plotname=plotdir+'2373_1.png', combined=False)
