---
title: "Ultrasonic image reconstruction using Synthetic Aperture Focusing Technique (SAFT)"
date: 2020-10-28
tags: [Ultrasonic, TFM, Image Processing, SAFT, Concrete, NDE]
header:
  image: "/images/perceptron/percept.jpg"
excerpt: "Ultrasonic, TFM, Image Processing, SAFT, Concrete, NDE"
mathjax: "true"
---


# Ultrasound Panoramic Imaging using Total Focusing Method
## Author: Sina Mehdinia








```python
# Reading data 

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time

t1 = time.time()
X = pd.read_excel('raw43.xlsx',header=None).values
t2 = time.time() - t1
print ('Data Loaded Successfully in',t2,'seconds')
```

    Data Loaded Successfully in 22.105371475219727 seconds
    


```python
# Full Matrix Capture (FMC)

def FMC(data, n):
  [row,col]=np.shape(data)
  tri_mat = np.zeros((col,n,n))
  for i in range(1,n):
    tri_mat[:,0,i]=data[i-1]
  for i in range(n,2*n-2):
    tri_mat[:,1,i-6]=data[i-1]
  for i in range(2*n-2, 3*n-5):
    tri_mat[:,2,i-11]=data[i-1]
  for i in range(3*n-5, 4*n-9):
    tri_mat[:,3,i-15]=data[i-1]
  for i in range(4*n-9, 5*n-14):
    tri_mat[:,4,i-18]=data[i-1]
  for i in range(5*n-14, 6*n-20):
    tri_mat[:,5,i-20]=data[i-1]
  for i in range(6*n-20, 7*n-27):
    tri_mat[:,6,i-21]=data[i-1]
  full_mat = tri_mat+ tri_mat.transpose(0,2,1)
  return full_mat 

# TFM Function (Equation 1)

def TFM(data,V,const,Fs,Nch,d,res,mesh_dim):
  xn = np.arange(0,mesh_dim[0]*res,res)
  yn = np.arange(0,mesh_dim[1]*res,res)
  xg,yg = np.meshgrid(yn,xn)
  yg_sq = yg**2 
  im = np.zeros((len(xn),len(yn)))
  for i in range(Nch):
    for j in range(Nch):
      t=np.round((((np.sqrt((xg-i*d+d)**2+yg_sq))+(np.sqrt((xg-j*d+d)**2+yg_sq)))/V+const)*Fs);  
      im += data[t.astype(int),i,j]
  return im
```


```python
# Main Script

Nch =  8
Y = FMC(X,Nch)
mesh_dim =[300,168]# Desired region of interest
num_scans=98;      # Number of Measurements
spacing=0.01;      # Overlapping measurements distance
Fs=10**6;          # Sampling Rate
Nch=8;             # Number of channels
d=0.03;            # Space between transducers
Res=0.00125;       # Desired Resolution
V=2300;            # Assuming wave speed (2200 for raw 47 data) (2300 for raw 43 data)
const = 0.00005;   # Assuming epsilon
image = []
for frame_num in range(num_scans):        # Reconstruction of all 98 Measurements
  X2=X[28*frame_num:28*frame_num+28,:]
  Y= FMC(X2,Nch)
  im = TFM(Y,V,const,Fs,Nch,d,Res,mesh_dim)
  image.append(im)

# Visualizing some random images from total of 98 images
fig1=plt.figure(figsize=(12, 12)) 
columns = 3
rows = 2
for i in range(1, columns*rows +1):
    rand_im = np.random.randint(1,99)-1
    img = image[rand_im]
    fig1.add_subplot(rows, columns, i)
    plt.imshow(img,cmap='gray')
    plt.xlabel('Length (m)', fontsize = 12)
    plt.ylabel('Height (m)', fontsize = 12)
    x_ticks = np.arange(0,168,80)
    y_ticks = np.arange(0,300,100)
    plt.yticks(y_ticks,y_ticks*Res, fontsize = 12)
    plt.xticks(x_ticks,x_ticks*Res, fontsize = 12)
plt.show()
```


![png](uea_tfm_files/uea_tfm_3_0.png)



```python
np.save('images', image)
```


```python
# Recosnruction of Panaromic image and final visualization

panaromic_image=np.zeros((mesh_dim[0],int(mesh_dim[1]+(num_scans-1)*(spacing/Res)+0.01/Res)));
k=0;
t1 = time.time()
for z in range(num_scans):
  for i in range(mesh_dim[0]):
    for j in range(k,mesh_dim[1]+k):
      panaromic_image[i,j+int(0.01/Res)] += image[z][i,j-k]
  k += int(spacing/Res)
t2 = time.time() - t1
print('Panaromic image reconstructed sucessfully in',t2,'seconds.')
fig2=plt.figure(figsize=(12, 12))
plt.imshow(panaromic_image,cmap= 'gray')
plt.xlabel('Length (m)', fontsize = 14)
plt.ylabel('Height (m)', fontsize = 14)
x_ticks = np.arange(0,1000,100)
y_ticks = np.arange(0,350,100)
plt.yticks(y_ticks,y_ticks*Res, fontsize = 14)
plt.xticks(x_ticks,x_ticks*Res, fontsize = 14)
plt.show()
```

    Panaromic image reconstructed sucessfully in 3.6144394874572754 seconds.
    


![png](uea_tfm_files/uea_tfm_5_1.png)
