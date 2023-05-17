#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 23:39:31 2022

@author: lisandro
"""

def P1_plus(u, P_A, P_B, dt):
    u1 = P_A(u, dt)
    u2 = P_B(u1, dt)
    return u2

def P1_minus(u, P_A, P_B, dt):
    u1 = P_B(u, dt)
    u2 = P_A(u1, dt)
    return u2

def P2_plus(u, P_A, P_B, dt):
    u1 = P1_plus(u, P_A, P_B, dt)
    u2 = P1_plus(u1,P_A, P_B, dt)
    return u2

def P2_minus(u, P_A, P_B, dt):
    u1 = P1_minus(u, P_A, P_B, dt)
    u2 = P1_minus(u1, P_A, P_B, dt)
    
    return u2

def P3_plus(u, P_A, P_B, dt):
    u1 = P2_plus(u, P_A, P_B, dt)
    u2 = P1_plus(u1,P_A, P_B, dt)
    return u2

def P3_minus(u, P_A, P_B, dt):
    u1 = P2_minus(u, P_A, P_B, dt)
    u2 = P1_minus(u1, P_A, P_B, dt)
    return u2

def P4_plus(u, P_A, P_B, dt):
    u1 = P3_plus(u, P_A, P_B, dt)
    u2 = P1_plus(u1, P_A, P_B, dt)
    return u2

def P4_minus(u, P_A, P_B, dt):
    u1 = P3_minus(u, P_A, P_B, dt)
    u2 = P1_minus(u1, P_A, P_B, dt)
    return u2

def affineS2_step(u, P_A, P_B, dt):
    gamma_1 = 1/2.
    
    u = gamma_1 * (P1_plus(u, P_A, P_B, dt) + P1_minus(u, P_A, P_B, dt))
    return u

def affineS4_step(u, P_A, P_B, dt):
    
    gamma_1 = -1/6.
    gamma_2 = 2/3.
    
    u = gamma_1 * (P1_plus(u, P_A, P_B, dt) + P1_minus(u, P_A, P_B, dt))+ \
        gamma_2 * (P2_plus(u, P_A, P_B, dt/2) + P2_minus(u, P_A, P_B, dt/2))
        
    return u

def affineS6_step(u, P_A, P_B, dt):
        
    gamma_1 = 1/48.
    gamma_2 = -8/15.
    gamma_3 = 81/80.

    u = gamma_1 * (P1_plus(u, P_A, P_B, dt) + P1_minus(u, P_A, P_B, dt)) + \
        gamma_2 * (P2_plus(u, P_A, P_B, dt/2) + P2_minus(u, P_A, P_B, dt/2))+\
        gamma_3 * (P3_plus(u, P_A, P_B, dt/3) + P3_minus(u, P_A, P_B, dt/3))

    return u

def affineS8_step(u, P_A, P_B, dt):
        
    gamma_1 = -1/720.
    gamma_2 = 8/45.
    gamma_3 = -729/560.
    gamma_4 = 512/315.
    
    u = gamma_1 * (P1_plus(u, P_A, P_B, dt) + P1_minus(u,P_A, P_B, dt)) +\
        gamma_2 * (P2_plus(u, P_A, P_B, dt/2) + P2_minus(u, P_A, P_B, dt/2)) +\
        gamma_3 * (P3_plus(u, P_A, P_B, dt/3) + P3_minus(u, P_A, P_B, dt/3)) +\
        gamma_4 * (P4_plus(u, P_A, P_B, dt/4) + P4_minus(u, P_A, P_B, dt/4))
        
    return u