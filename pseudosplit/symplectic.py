#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 23:50:11 2022

@author: lisandro
"""

def neri_step(u, P_A, P_B, dt):

    c1 = c4 = 1 / 2 / (2 - 2**(1 / 3)) * dt
    c2 = c3 = (1 - 2 ** (1 / 3)) / 2 / (2 - 2**(1 / 3)) * dt
    d1 = d3 = 1 / (2 - 2**(1 / 3)) * dt
    d2 = -2 ** (1 / 3) / (2 - 2 ** (1 / 3)) * dt
    d4 = 0.

    u1 = P_A(u, c1)
    u2 = P_B(u1, d1)
    u3 = P_A(u2, c2)
    u4 = P_B(u3, d2)
    u5 = P_A(u4, c3)
    u6 = P_B(u5, d3)
    u7 = P_A(u6, c4)

    return u7

def strang_step(u, P_A, P_B, dt):
    u1 = P_A(u, dt/2)
    u2 = P_B(u1, dt)
    u3 = P_A(u2, dt/2)
    return u3

def ruth_step(u, P_A, P_B, dt):

    c1 = 1. * dt
    c2 = - 2 / 3. * dt
    c3 = 2 / 3. * dt
    d1 = - 1 / 24. * dt
    d2 = 3 / 4. * dt
    d3 = 7 / 24. * dt

    u1 = P_A(u, c1)
    u2 = P_B(u1, d1)
    u3 = P_A(u2, c2)
    u4 = P_B(u3, d2)
    u5 = P_A(u4, c3)
    u6 = P_B(u5, d3)

    return u6

def yoshida_6_step(u, P_A, P_B, dt):
    a0 = 0.392256805238778632 * dt
    a1 = 0.510043411918457699 * dt
    a2 = -0.471053385409756437 * dt
    a3 = 0.068753168252520106 * dt
    a4 = 0.068753168252520106 * dt
    a5 = -0.471053385409756437 * dt
    a6 = 0.510043411918457699 * dt
    a7 = 0.392256805238778632 * dt
    b0 = 0.784513610477557264 * dt
    b1 = 0.235573213359358134 * dt
    b2 = -1.177679984178871007 * dt
    b3 = 1.315186320683911218 * dt
    b4 = -1.177679984178871007 * dt
    b5 = 0.235573213359358134 * dt
    b6 = 0.784513610477557264 * dt
    b7 = 0.000000000000000000 * dt

    u1 = P_B(P_A(u, a0), b0)
    u2 = P_B(P_A(u1, a1), b1)
    u3 = P_B(P_A(u2, a2), b2)
    u4 = P_B(P_A(u3, a3), b3)
    u5 = P_B(P_A(u4, a4), b4)
    u6 = P_B(P_A(u5, a5), b5)
    u7 = P_B(P_A(u6, a6), b6)
    u8 = P_A(u7, a7)

    return u8
