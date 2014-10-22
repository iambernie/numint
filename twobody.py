#!/usr/bin/env python3

import h5py
import numpy
from numpy import sin, cos, pi, degrees
from ext import hdf5handler
from matplotlib import pyplot as plt

#MKS
G = 6.67384e-11       # m^3 kg^-1 s^-2
MSun = 1.9891e30      # kg^1
AU = 149597870700     # m^1
DAY = 3600*24         # s^1
YEAR = DAY*365.25     # s^1

def rk4(h, f, t, z):
    k1 = h * f(t, z)
    k2 = h * f(t + 0.5*h, z + 0.5*k1)
    k3 = h * f(t + 0.5*h, z + 0.5*k2)
    k4 = h * f(t + h, z + k3)
    return z + (k1 + 2*(k2 + k3) + k4)/6.0

def forward_euler(h, f, t, z):
    return z + h*f(t, z)

def twobody_vmass(t, state):
    a, e, f, w, M = state

    n = (G*M/a**3)**(1/2) #mean motion

    dM = -1e-5*MSun/YEAR  #implement here?
    da =  -a*(1 + e**2 + 2*e*cos(f)) / (1-e**2) * dM/M
    de =  -(e+cos(f)) * dM/M
    dw =  -sin(f) / e * dM/M
    df =  -dw + n*(1+e*cos(f))**2 / ((1 - e**2)**(3/2))
    return numpy.array([da, de, df, dw, dM])


def main():
    #todo:energy, ang_mom, position, etc.

    dt = 1*YEAR
    M0 = 1*MSun
    a0 = 100*AU
    e0 = 0.9
    f0 = 0
    w0 = 0
    state = numpy.array([a0, e0, f0, w0, M0])
    with hdf5handler.HDF5Handler('test.hdf5') as handle:
        for method in [rk4]:
            handle.prefix = method.__name__
            for t in numpy.arange(0, 5e4*YEAR, dt):
                print("t={:.3f} year".format(t/YEAR))
                state = method(dt, twobody_vmass, t, state)
                handle.put(t, '/time')
                handle.put(state[0], '/a')
                handle.put(state[1], '/e')
                handle.put(state[2], '/f')
                handle.put(state[3], '/w')
                handle.put(state[4], '/mass')


    f = h5py.File('test.hdf5')
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    for method in ['rk4']:
        ax1.plot(f[method+'/time'].value/YEAR , f[method+'/a'].value / AU)
        ax2.plot(f[method+'/time'].value/YEAR , f[method+'/e'])
        ax3.plot(f[method+'/time'].value/YEAR , f[method+'/f'].value % (2*pi))
        ax4.plot(f[method+'/time'].value/YEAR , f[method+'/mass'].value / MSun)

    plt.savefig('image.png')


    fig = plt.figure()
    ax = fig.add_subplot(111,polar=True)
    for method in ['rk4']:
        ax.plot(f[method+'/f'].value %(2*pi), f[method+'/time'].value /YEAR )
    plt.savefig('image1.png')


    fig = plt.figure()
    ax = fig.add_subplot(111,polar=True)
    for method in ['rk4']:
        ax.plot(f[method+'/w'].value %(2*pi), f[method+'/time'].value /YEAR )
    plt.savefig('image2.png')

if __name__ == "__main__":
    main()


