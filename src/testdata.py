# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import pandas as pd


# Mobius strip
S = 50
T = 10
sigma = 0
s = np.linspace(0.0, 4 * np.pi, S)[None, :]
t = np.linspace(-1.0, 1.0, T)[:, None]
x = (1 + 0.5 * t * np.cos(0.5 * s)) * np.cos(s)
y = (1 + 0.5 * t * np.cos(0.5 * s)) * np.sin(s)
z = 0.5 * t * np.sin(0.5 * s)
P = np.stack([x, y, z], axis=-1)
data = pd.DataFrame(P.reshape(S*T,-1)  + sigma*np.random.randn(S*T,3))
data = pd.merge(index,data,left_index=True,right_index=True)
data = data.set_index(['orientation', 'contrast', 'frequency', 'phase'])

# Klein bottle
N = 30
sigma = 0.05
A = 4
B = 1
theta = np.linspace(0.0, 2 * np.pi, N)[None, :]
v = np.linspace(0.0, 2 * np.pi, 2*N)[:, None]
x = A*(np.cos(theta/2)*np.cos(v) - np.sin(theta/2)*np.sin(2*v))
y = A*(np.sin(theta/2)*np.cos(v) + np.cos(theta/2)*np.sin(2*v))
z = B*np.cos(theta)*(1 + 0.1*np.sin(v))
w = B*np.sin(theta)*(1 + 0.1*np.sin(v))
P = np.stack([x, y, z, w], axis=-1)
data = pd.DataFrame(P.reshape(2*N**2,-1)  + sigma*np.random.randn(2*N**2,4))
data = pd.merge(index,data,left_index=True,right_index=True)
data = data.set_index(['orientation', 'contrast', 'frequency', 'phase'])