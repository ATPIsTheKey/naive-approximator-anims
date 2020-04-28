#!/usr/local/bin/python
# -*- coding: ascii -*-

import numpy as np
import scipy.special as sc_spec

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

import time
from tqdm import tqdm

""" Configure animation and matplotlib """

mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

ANIMATION_LEN = 2  # [s]
ANIMATION_FPS = 30
ANIMATION_DPI = 300
SAME_ANGFREQ = (True, 10)  # change this to (False, 0) to let membranes oscillate at eigenfrequency
PATH_OUT = ''  # specify path where animation should be saved


""" Physical constants: change these to modify behaviour of membrane in animation """

MEMBRANE_CENTER = (0.5, 0.5, 0.5)  # (x, y, z)
MEMBRANE_RADIUS = 0.5  # [m]
MEMBRANE_THICKNESS = 0.01  # [m]
MEMBRANE_DENSITY = 0.86  # density of dry leather [kg/m^3]
RADIAL_MEMBRANE_RESULTANT = 0.2  # randomly chosen [N]

A, B, C, D = 1, 1, 1, 1  # randomly chosen
C_SMALL = np.sqrt(RADIAL_MEMBRANE_RESULTANT / (MEMBRANE_THICKNESS * MEMBRANE_DENSITY))  # [m/s]

# taken from http://wwwal.kuicr.kyoto-u.ac.jp/www/accelerator/a4/besselroot.htmlx
FIRST_ORDER_BESSEL_ROOTS = (
    (2.40482555769577, 	5.52007811028631, 	8.65372791291101, 	11.7915344390142, 	14.9309177084877),
    (3.83170597020751, 	7.01558666981561, 	10.1734681350627, 	13.3236919363142, 	16.4706300508776),
    (5.13562230184068, 	8.41724414039986, 	11.6198411721490, 	14.7959517823512, 	17.9598194949878),
    (6.38016189592398, 	9.76102312998166, 	13.0152007216984, 	16.2234661603187, 	19.4094152264350),
    (7.58834243450380, 	11.0647094885011, 	14.3725366716175, 	17.6159660498048, 	20.8269329569623),
    (9.93610952421768, 	13.5892901705412, 	17.0038196678160, 	20.3207892135665, 	23.5860844355813),
    (11.0863700192450, 	14.8212687270131, 	18.2875828324817, 	21.6415410198484, 	24.9349278876730),
    (12.2250922640046, 	16.0377741908877, 	19.5545364309970, 	22.9451731318746, 	26.2668146411766),
    (13.3543004774353, 	17.2412203824891, 	20.8070477892641, 	24.2338852577505, 	27.5837489635730),
    (14.4755006865545, 	18.4334636669665, 	22.0469853646978, 	25.5094505541828, 	28.8873750635304)
)


""" 
Functions used to solve the cyclindrical wave equation; all information taken from 
- https://en.wikipedia.org/wiki/Wave_equation
- https://en.wikipedia.org/wiki/Vibrations_of_a_circular_membrane 
- https://en.wikipedia.org/wiki/Bessel_function
"""

def lambda_mn(m: int, n: int) -> float:
    return FIRST_ORDER_BESSEL_ROOTS[m][n - 1] / MEMBRANE_RADIUS


def solution_u_mn(r: float, phi: float, t: float, m: int, n: int) -> float:
    lmn = lambda_mn(m, n)
    w = SAME_ANGFREQ[1] if SAME_ANGFREQ[0] else lmn * C_SMALL
    return (A * np.cos(w * t) + B * np.sin(w * t)) * \
           sc_spec.jv(m, lmn * r) * (C * np.cos(m * phi) + D * np.sin(m * phi))


""" Main function initialising the solution and animation """

def animation_main():
    # compute surface coordinates of a circular membrane in cylindrical coordinates
    dr = 1e-2
    dphi = 1e-2
    dz = 1e-2

    rho_m = MEMBRANE_RADIUS
    phi0, phi_m = 0, 2 * np.pi
    z0, z_m = 0, 0  # ignore thickness of membrane in plot

    m = int(np.ceil(max((rho_m / dr, (phi_m - phi0) / dphi, (z_m - z0) / dz))))  # compute minimum number of vals necessary for linspace
    rho, phi = np.linspace(0, rho_m, m), np.linspace(phi0, phi_m, m)
    rho_grid, phi_grid = np.meshgrid(rho, phi)

    # convert cylindrical coordinates to cartesian coordinates
    Xc, Yc = MEMBRANE_CENTER[0] + np.cos(phi_grid) * rho_grid, MEMBRANE_CENTER[1] + np.sin(phi_grid) * rho_grid

    # compute vectors of solutions in time
    dt = 1e-2
    t_max = ANIMATION_LEN   # specify length of animation here
    m1, n1 = 0, 1  # specify the modes here
    m2, n2 = 0, 3
    m3, n3 = 1, 1
    m4, n4 = 1, 3
    m5, n5 = 2, 1
    m6, n6 = 2, 3

    print('Begin computing solutions')
    t0 = time.time()
    pbar = tqdm(total=6)
    Zsols1 = np.array(
        [solution_u_mn(rho_grid, phi_grid, t, m1, n1) for t in np.arange(0, t_max, dt)])  # numpy is sick
    pbar.update(1)
    Zsols2 = np.array(
        [solution_u_mn(rho_grid, phi_grid, t, m2, n2) for t in np.arange(0, t_max, dt)])
    pbar.update(1)
    Zsols3 = np.array(
        [solution_u_mn(rho_grid, phi_grid, t, m3, n3) for t in np.arange(0, t_max, dt)])
    pbar.update(1)
    Zsols4 = np.array(
        [solution_u_mn(rho_grid, phi_grid, t, m4, n4) for t in np.arange(0, t_max, dt)])
    pbar.update(1)
    Zsols5 = np.array(
        [solution_u_mn(rho_grid, phi_grid, t, m5, n5) for t in np.arange(0, t_max, dt)])
    pbar.update(1)
    Zsols6 = np.array(
        [solution_u_mn(rho_grid, phi_grid, t, m6, n6) for t in np.arange(0, t_max, dt)])
    pbar.update(1)
    tf = time.time()
    pbar.close()
    delta_t = tf - t0
    print(f'Finished computing solutions in {int(delta_t // 3600)}h:{int(delta_t / 60)}m:{int(delta_t % 60)}s')

    # needed for colormap
    vmin1, vmax1 = np.amin(Zsols1.flat), np.amax(Zsols1.flat)
    vmin2, vmax2 = np.amin(Zsols1.flat), np.amax(Zsols1.flat)
    vmin3, vmax3 = np.amin(Zsols1.flat), np.amax(Zsols1.flat)
    vmin4, vmax4 = np.amin(Zsols1.flat), np.amax(Zsols1.flat)
    vmin5, vmax5 = np.amin(Zsols1.flat), np.amax(Zsols1.flat)
    vmin6, vmax6 = np.amin(Zsols1.flat), np.amax(Zsols1.flat)

    # setup the plot
    fig = plt.figure(figsize=(7, 6))
    fig.tight_layout()
    plt.subplots_adjust(bottom=0, wspace=0.3, top=0.85)
    fig.suptitle(r"\textbf{Eigenmodes of a circular membrane}", fontsize=17, y=0.95)

    ax1 = fig.add_subplot(3, 2, 1, projection='3d')
    ax2 = fig.add_subplot(3, 2, 2, projection='3d')
    ax3 = fig.add_subplot(3, 2, 3, projection='3d')
    ax4 = fig.add_subplot(3, 2, 4, projection='3d')
    ax5 = fig.add_subplot(3, 2, 5, projection='3d')
    ax6 = fig.add_subplot(3, 2, 6, projection='3d')

    def plot(i):  # callback function for FuncAnimation
        ax1.cla()
        ax2.cla()
        ax3.cla()
        ax4.cla()
        ax5.cla()
        ax6.cla()

        ax1.set_title(f'mode ${m1}-{n1}$')
        ax1.set_axis_off()
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_zlim(-2, 2)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.view_init(58, -56)

        ax2.set_title(f'mode ${m2}-{n2}$')
        ax2.set_axis_off()
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_zlim(-2, 2)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.view_init(58, -56)

        ax3.set_title(f'mode ${m3}-{n3}$')
        ax3.set_axis_off()
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_zlim(-2, 2)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.view_init(58, -56)

        ax4.set_title(f'mode ${m4}-{n4}$')
        ax4.set_axis_off()
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax4.set_zlim(-2, 2)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.view_init(58, -56)

        ax5.set_title(f'mode ${m5}-{n5}$')
        ax5.set_axis_off()
        ax5.set_xticks([])
        ax5.set_yticks([])
        ax5.set_zlim(-2, 2)
        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)
        ax5.view_init(58, -56)

        ax6.set_title(f'mode ${m6}-{n6}$')
        ax6.set_axis_off()
        ax6.set_xticks([])
        ax6.set_yticks([])
        ax6.set_zlim(-2, 2)
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.view_init(58, -56)

        Zc1 = Zsols1[i] + MEMBRANE_CENTER[2]  # offset Z solutions to center of membrane
        Zc2 = Zsols2[i] + MEMBRANE_CENTER[2]  # offset Z solutions to center of membrane
        Zc3 = Zsols3[i] + MEMBRANE_CENTER[2]  # offset Z solutions to center of membrane
        Zc4 = Zsols4[i] + MEMBRANE_CENTER[2]  # offset Z solutions to center of membrane
        Zc5 = Zsols5[i] + MEMBRANE_CENTER[2]  # offset Z solutions to center of membrane
        Zc6 = Zsols6[i] + MEMBRANE_CENTER[2]  # offset Z solutions to center of membrane

        ax1.plot_surface(Xc, Yc, Zc1, alpha=0.8, cmap='jet', lw=0, rcount=100, ccount=100,
                         vmin=vmin1, vmax=vmax1)
        ax2.plot_surface(Xc, Yc, Zc2, alpha=0.8, cmap='jet', lw=0, rcount=100, ccount=100,
                         vmin=vmin2, vmax=vmax2)
        ax3.plot_surface(Xc, Yc, Zc3, alpha=0.8, cmap='jet', lw=0, rcount=100, ccount=100,
                         vmin=vmin3, vmax=vmax3)
        ax4.plot_surface(Xc, Yc, Zc4, alpha=0.8, cmap='jet', lw=0, rcount=100, ccount=100,
                         vmin=vmin4, vmax=vmax4)
        ax5.plot_surface(Xc, Yc, Zc5, alpha=0.8, cmap='jet', lw=0, rcount=100, ccount=100,
                         vmin=vmin5, vmax=vmax5)
        ax6.plot_surface(Xc, Yc, Zc6, alpha=0.8, cmap='jet', lw=0, rcount=100, ccount=100,
                         vmin=vmin6, vmax=vmax6)

        pbar.update(1)

    # setup the animation
    n_frames = int(np.ceil(t_max / dt))
    anim = FuncAnimation(
        fig, plot,
        frames=n_frames,
        interval=1000 / ANIMATION_FPS,  # delay between frames in milliseconds
        repeat=True
    )

    print('Begin rendering')
    pbar = tqdm(total=n_frames)
    t0 = time.time()
    anim.save(
        PATH_OUT + 'circular_membrane_anim.mp4',
        writer='ffmpeg', dpi=ANIMATION_DPI
    )
    pbar.close()
    tf = time.time()
    delta_t = tf - t0
    print(f'Finished rendering {n_frames} frames in '
          f'{int(delta_t // 3600)}h:{int(delta_t // 60)}m:{int(delta_t % 60)}s')


if __name__ == '__main__':
    animation_main()
