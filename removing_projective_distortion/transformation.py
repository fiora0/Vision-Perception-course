# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 18:31:14 2020

@author: fiora
"""

## To be used only with run from console 
import matplotlib
matplotlib.use("TkAgg")
##

import sys
import os

from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import scipy
from scipy import linalg, matrix
from scipy.interpolate import  griddata



def load_image(im):  
    try:
        imx = Image.open(im)
        return np.array(imx)
        # else
    except IOError:
        return []
    
def select_four_points(im):       
     plt.imshow(im)     
     xx = plt.ginput(4)
     xx.append(xx[0])
     return xx
     
     
def generate_GT(xx):
    plt.figure(2)
 
    zzd = np.zeros((5,3))   
    for ii in range(len(xx)-1):         
#        x2 =xx[ii+1][0]; y2=xx[ii+1][1]
        x1 =xx[ii][0]; y1=xx[ii][1]
        zzd[ii,0] = x1; zzd[ii,1] = y1; zzd[ii,2] = 1; 
        plt.plot([xx[ii][0],xx[ii+1][0]], [xx[ii][1],xx[ii+1][1]], 'ro-') 
    jj = 0
    aa = [0,0,1,0,1,3,0,3]
    zz = np.zeros((5,3))     
    for ii in range(len(zzd)-1):
            zz[ii,0] = zzd[aa[jj],0] 
            zz[ii,1] = zzd[aa[jj+1],1] 
            zz[ii,2] = 1;   
            jj = jj+2
    zz[4,:] = zz[0,:]
    for ii in range(4):      
        plt.plot([zz[ii,0],zz[ii+1,0]], [zz[ii,1],zz[ii+1,1]], 'go-')
    plt.show()
    return zz[0:4,:],zzd[0:4,:]    
        
    
#We have x' = z and x =zd
def normalize_points(zz):
    uu = zz.T
    ff_xx = np.ones(uu.shape)
    indices, = np.where(abs(uu[2,:]) > 10**-12)
    ff_xx[0:2,indices] = uu[0:2,indices]/uu[2,indices]
    ff_xx[2,indices]  = 1.
    mu = np.mean(ff_xx[0:2,:],axis = 1)
    mu_r = np.zeros((mu.shape[0],ff_xx.shape[1]))
    for ii in range(ff_xx.shape[1]):
        mu_r[:,ii] = mu
    mu_dist = np.mean((np.sum((ff_xx[0:2] - mu_r)**2,axis =0))**0.5)
    scale =  (2**0.5/mu_dist)
    s0 = -scale*mu[0]
    s1 = -scale*mu[1]
    S = np.array([[scale, 0, s0],[0, scale, s1], [0, 0, 1]])
    normalized_zz = S@ff_xx
    return normalized_zz, S
    
def compute_A(uu,vv):
    #build for i =1:4 the matrix Ai
    ## uu=GT = x' in HZ, vv= distorted = x in HZ
    A = np.zeros((2*(uu.shape[0]+1),9))
    jj = 0
    for ii in range(uu.shape[0]+1):
        a = (np.zeros((1,3))[0] )       
        b = (-uu[2,ii] * vv[:,ii]) 
        c =  uu[1,ii] * vv[:,ii]
        d =  uu[2,ii] * vv[:,ii]
        f =  (-uu[0,ii]*vv[:,ii])
        row1 = np.concatenate((a, b, c), axis=None)
        row2 = np.concatenate((d,a,f), axis=None)
        A[jj,:] = row1
        A[jj+1,:] = row2
        jj = jj+2
    return A


def compute_homography(A,T1,T2):
#
    null_space_of_A = -scipy.linalg.null_space(A)
    hh_normalized = np.reshape(null_space_of_A,(3,3)) 
    hh = np.dot(np.linalg.inv(T2),np.dot(hh_normalized,T1))
    return hh

### Transform Image
    
def rgb2gray(imm):
    return np.dot(imm[...,:3], [0.2989, 0.5870, 0.1140])

def image_rebound(mm,nn,hh):
    W = np.array([[1, nn, nn, 1 ],[1, 1, mm, mm],[ 1, 1, 1, 1]])
    ws = np.dot(hh,W)
    ### scaling
    xx = np.vstack((ws[2,:],ws[2,:],ws[2,:]))
    wsX =  np.round(ws/xx)
    bounds = [np.min(wsX[1,:]), np.max(wsX[1,:]),np.min(wsX[0,:]), np.max(wsX[0,:])]
    return bounds


def make_transform(imm,hh):   
    mm,nn = imm.shape[0],imm.shape[0]
    bounds = image_rebound(mm,nn,hh)
    nrows = bounds[1] - bounds[0]
    ncols = bounds[3] - bounds[2]
    s = max(nn,mm)/max(nrows,ncols)
    scale = np.array([[s, 0, 0],[0, s, 0], [0, 0, 1]])
    trasf = scale@hh
    trasf_prec =  np.linalg.inv(trasf)
    bounds = image_rebound(mm,nn,trasf)
    nrows = (bounds[1] - bounds[0]).astype(np.int)
    ncols = (bounds[3] - bounds[2]).astype(np.int)
    return bounds, nrows, ncols, trasf, trasf_prec


def get_new_image(nrows,ncols,imm,bounds,trasf_prec,nsamples):
    xx  = np.linspace(1, ncols, ncols)
    yy  = np.linspace(1, nrows, nrows)
    [xi,yi] = np.meshgrid(xx,yy) 
    a0 = np.reshape(xi, -1,order ='F')+bounds[2]
    a1 = np.reshape(yi,-1, order ='F')+bounds[0]
    a2 = np.ones((ncols*nrows))
    uv = np.vstack((a0.T,a1.T,a2.T)) 
    new_trasf = np.dot(trasf_prec,uv)
    val_normalization = np.vstack((new_trasf[2,:],new_trasf[2,:],new_trasf[2,:]))
   
    ### The new transformation
    newT = new_trasf/val_normalization
    
    ### 
    xi = np.reshape(newT[0,:],(nrows,ncols),order ='F') 
    yi = np.reshape(newT[1,:],(nrows,ncols),order ='F')
    cols = imm.shape[1]
    rows = imm.shape[0]
    xxq  = np.linspace(1, rows, rows).astype(np.int)
    yyq  = np.linspace(1, cols, cols).astype(np.int)
    [x,y] = np.meshgrid(yyq,xxq) 
    x = (x - 1).astype(np.int) #Offset x and y relative to region origin.
    y = (y - 1).astype(np.int) 
        
    ix = np.random.randint(im.shape[1], size=nsamples)
    iy = np.random.randint(im.shape[0], size=nsamples)
    samples = im[iy,ix]
       
    int_im = griddata((iy,ix), samples, (yi,xi))
    
    #Plotting
    fig = plt.figure(figsize=(8, 8))
    columns = 2
    rows = 1
    fig.add_subplot(rows, columns, 1)
    plt.imshow(im)
    fig.add_subplot(rows, columns, 2)
    
    plt.imshow(int_im.astype(np.uint8))
    plt.show()
   
    
  
images_available =["campus.jpg",  "hilton.png", "buco.jpg", "hotel.jpg","casaconOcclusion.jpg","palazzo.jpg", "build0.jpg"]
 
    
### Main
if __name__ == "__main__":
    imnum= sys.argv[1]
    cwd = os.getcwd()
    ddir = os.listdir(cwd)
    if not 'dataset' in ddir:
        print(('oops there is no dataset folder') )
    else:  
        image_building = images_available[int(float(imnum))]
        imagetoload = os.path.join(cwd,'dataset',image_building)
        im =  load_image(imagetoload)
        xx = select_four_points(im)
        zz, zzd = generate_GT(xx)
            
        ## Normalize points and return also scaling matrix
        norm_points_distorted, T1_norm = normalize_points(zzd)
        norm_points_GT, T2_norm= normalize_points(zz)
        
        
        A = compute_A(norm_points_GT,norm_points_distorted)
        hh =  compute_homography(A,T1_norm,T2_norm)
        
        ### 
        bounds, nrows, ncols,  trasf, trasf_prec = make_transform(im,hh)     
        nn,mm  = im.shape[0],im.shape[0]
        if max(nn,mm)>1000:
            kk =6
        else: kk =5
        nsamples = 10**kk         
        get_new_image(nrows,ncols,im,bounds,trasf_prec,nsamples)
    
## Transform image
