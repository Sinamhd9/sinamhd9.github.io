---
title: "pile"
date: 2021-01-16
tags: [computer vision, image processing, panorama, ransac, opencv]
header:
  image: "/images/image_stitch/panorama.jfif"
excerpt: "Computer vision, Image processing, Panorama, RANSAC, OpenCV"
mathjax: "true"
---


```python
%matplotlib widget
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import FloatSlider,BoundedFloatText, BoundedIntText, IntSlider, Layout, VBox, HBox, Output, Label
from IPython.display import clear_output
```


```python
def stiff_rel_fun(E, I, L):
    k = np.array([[12*E*I/L**3, 6*E*I/L**2, -12*E*I/L**3, 6*E*I/L**2],
    [6*E*I/L**2, 4*E*I/L, -6*E*I/L**2, 2*E*I/L],
    [-12*E*I/L**3, -6*E*I/L**2, 12*E*I/L**3, -6*E*I/L**2],
    [6*E*I/L**2, 2*E*I/L, -6*E*I/L**2, 4*E*I/L]], dtype=np.float64)
    return k
```


```python
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10,5), sharey=True)
```


```python
style = {'description_width': 'initial', 'handle_color': 'lightgreen'}
d_slider = FloatSlider(
    value=0.6,
    min=0.1,
    max=2.0,
    step=0.1,
    description='Pile diameter ($m$):',style=style,layout=Layout(width='500px')
)
facw_slider = FloatSlider(
    value=0.8,
    min=0.0,
    max=1.0,
    step=0.1,
    description='Fraction of section width in spring stiffness:',style=style,
    layout=Layout(width='500px')
)

L_slider = FloatSlider(
    value=31.5,
    min=0.5,
    max=50.0,
    step=0.1,
    description='Pile length ($m$):',style=style,
    layout=Layout(width='500px')
)
N_slider = IntSlider(
    value=100,
    min=10,
    max=1000,
    step=10,
    description='Number of elements',
    style=style,
    layout=Layout(width='500px')
)

k_slider = IntSlider(
    value=5800,
    min=1000,
    max=10000,
    step=100,
    description='Soil stiffness ($kN/m^2$)',
    style=style,
    layout=Layout(width='500px')
)


E_btb = BoundedIntText(
 value=2e8,
 min=1e7,
 max=1e10,
    step=1e6,
 description='Modulus of elasticity ($kN/m^2$):',
    style=style,
    layout=Layout(width='300px')
 )

I_btb = BoundedFloatText(
 value=0.00187,
 min=0.00001,
 max=0.1,
step=0.001,
 description='Second moment of inertia ($m^4$):',
    style=style,
    layout=Layout(width='300px')
 )


H_btb = BoundedIntText(
 value=200,
 min=1,
 max=1000,
 description='Lateral load at pile top ($kN$)',
    style=style,
    layout=Layout(width='300px')
 )


Ktheta_btb = BoundedIntText(
 value=1e8,
 min=0,
 max=1e12,
step=1e6,
 description='Rotational spring on pile top ($kNm/rad$)',
    style=style,
    layout=Layout(width='300px')
 )


ui = VBox([d_slider, facw_slider, L_slider, N_slider,  k_slider])
ui2 = VBox([E_btb, I_btb, H_btb, Ktheta_btb])
uis = HBox([ui, ui2])
display(uis)
```


```python
def run_calcs():
    d =  d_slider.value  
    facw = facw_slider.value
    L = L_slider.value
    E = E_btb.value                       
    I = I_btb.value
    N = N_slider.value
    H = H_btb.value
    k = k_slider.value
    Ktheta = Ktheta_btb.value 
    delta_x = L/N                               # Length of one element (in)
    K = stiff_rel_fun(E, I, delta_x).astype(np.float64)
    x = np.arange(0,L+0.01,delta_x, dtype=np.float64)                   # x-vector (in)
    u_y = np.zeros((len(x)), dtype=np.float64)
    F_u = np.zeros((2*(N+1), 1), dtype=np.float64)
    F_u[0] = H
    k_S = np.zeros((2*(N+1)), dtype=np.float64)
    k_S[0] = k*delta_x*facw*d/2
    k_S[2:-3:2] = k*delta_x*facw*d
    k_S[-2] = k*delta_x*facw*d/2
    k_S[1] = Ktheta

    # Create stiffness matrix
    K_uu = np.zeros((2*(N+1), 2*(N+1)), dtype=np.float64)
    K_uu[:2, :2] = K[:2, :2]
    K_uu[:2, 2:4] = K[:2, 2:]
    K_uu[0, 0] += k_S[0]
    K_uu[1, 1] += k_S[1]
    j = 2
    for _ in range(1,N):
        K_uu[j:j+2, j-2:j] = K[2:, :2]
        K_uu[j:j+2, j:j+2] = K[2:, 2:]+K[:2, :2]
        K_uu[j, j] += k_S[j]
        K_uu[j+1, j+1] += k_S[j+1]
        K_uu[j:j+2, j+2:j+4] = K[:2, 2:]
        j += 2
    K_uu[-2:, -4:-2] = K[2:, :2]
    K_uu[-2:, -2:] = K[2:, 2:]
    K_uu[-2, -2] += k_S[-2]
    K_uu[-1, -1] += k_S[-1]
    delta_u = np.linalg.solve(K_uu, F_u)
    Q = np.zeros((4, N), dtype=np.float64)
    M = np.zeros_like(x, dtype=np.float64)
    V = np.zeros_like(x, dtype=np.float64)
    j = 0
    for i in range(N):
        Q[:, i] = (K@(np.array([delta_u[j],-delta_u[j+1],-delta_u[j+2],delta_u[j+3]], dtype=np.float64))).reshape(4,)
        j = j+2
    V[:-1] = Q[0,:]
    M[:-1] = Q[1,:]
    V[-1] = Q[2,-1]
    M[-1] = Q[3,-1]
    u_y = delta_u[::2] * 1000
    return u_y, V, M, x, L

def visualize(u_y, V, M, x, L):
    axes[0].clear()
    axes[1].clear()
    axes[2].clear()
    axes[0].plot(u_y-0.15*max(abs(u_y)), x, 'k')
    axes[0].plot(u_y, x,'k')
    axes[0].plot(u_y+0.15*max(abs(u_y)), x, 'k')
    axes[0].plot([u_y[0]-0.15*max(abs(u_y)), u_y[0]+0.15*max(abs(u_y))], [x[0], x[0]], 'k')
    axes[0].plot([u_y[-1]-0.15*max(abs(u_y)), u_y[-1]+0.15*max(abs(u_y))], [x[-1], x[-1]], 'k')
    axes[0].plot(np.zeros((len(x))), x, 'k')
    axes[0].axis([1.01*min(u_y-0.25*max(abs(u_y)))-2,1.01*max(u_y+0.25*max(abs(u_y)))+2,0, L+1])
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Horizontal deformation, u_x (mm)')
    axes[0].set_ylabel('Depth, h (m)')
    axes[1].plot(np.zeros((len(x))), x, 'k')
    axes[1].fill_between(V, x, color='b')  
    axes[1].axis([1.01*min(V)-5, 1.01*max(V)+5, 0, L+1])
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Shear force, V (kN)')
    axes[1].set_ylabel('Depth, h (m)')
    axes[2].plot(np.zeros((len(x))), x, 'k')
    axes[2].fill_between(M, x, color='r')
    axes[2].axis([1.01*min(M)-5, 1.01*max(M)+5, 0, L+1])
    axes[2].invert_yaxis()
    axes[2].set_ylabel('Depth, h (m)')
    axes[2].set_xlabel('Bending moment, M (kNm)')
```


```python
out=Output()
def on_value_change(change):
    u_y, V, M, x, L = run_calcs()
    with out:
        clear_output()
        visualize(u_y, V, M, x, L)

        
for i in uis.children:
    for j in i.children:
        j.observe(on_value_change, 'value')
out
```


```python

```
