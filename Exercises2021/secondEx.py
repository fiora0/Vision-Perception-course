"""
Created on Sun Jul  4 12:29:20 2021

@author: Fiora
"""
""" Exercise 2
    Affine and metric transformations
"""

import numpy as np
import numpy.matlib
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.pyplot import ginput
from matplotlib import rcParams
from skimage.transform import warp, AffineTransform, ProjectiveTransform
from skimage.color import rgb2gray 
import scipy
from scipy import linalg, matrix




def crop_imm(imms, ss):
    
    ttx, tty = np.where(imms[:,:,0]>0)
    mm, nn = np.max(ttx), np.max(tty)
    mu, nu = np.min(ttx), np.min(tty)
    left, top, right, bottom = mu, nu, nn, mm
    immx = Image.fromarray(((imms*255).astype(np.uint8)))
    im_cropped = immx.crop((left, top, right, bottom)) 
    # im_resh = im_cropped.resize((ss[0], ss[1]),  Image.ANTIALIAS)
    return im_cropped

def projToAffine(imm):
    """ Exercise 2 questions 1,2 and 3:
        select 8 points, use them to compute parallel lines and points at infinity.
        Use points at infinity to build line at infinity.
        Build the homography H mapping the projective immage to an affine one
    """
    
    ### pick up and draw the points to define the line at infinity
    plt.imshow(imm)
    plt.title('Choose 8 point: 2 points per line: parallel lines along two orthogonal dir.')
    l1p = plt.ginput(8,timeout=60, show_clicks=True)
    #### first line
    x_values = [l1p[0][0], l1p[1][0]]
    y_values = [l1p[0][1], l1p[1][1]]
    plt.plot(x_values, y_values,'r', label = "line 1", linewidth=2)
    
    
    #### second line
    x_values = [l1p[2][0], l1p[3][0]]
    y_values = [l1p[2][1], l1p[3][1]]
    plt.plot(x_values, y_values,'r', label = "line 2", linewidth=2)
    
    #### third line
    x_values = [l1p[4][0], l1p[5][0]]
    y_values = [l1p[4][1], l1p[5][1]]
    plt.plot(x_values, y_values,'b', label = "line 3", linewidth=2)
    #### fourth line
    x_values = [l1p[6][0], l1p[7][0]]
    y_values = [l1p[6][1], l1p[7][1]]
    plt.plot(x_values, y_values,'b', label = "line 4", linewidth=2)
    #
   
    ### pick up and draw the points to define the line at infinity
    
    """ compute the points at infinity """
    lp = np.array(l1p,dtype= float)
    lp = np.column_stack([lp, np.matlib.repmat(1, len(lp),1)])
    l1 = np.cross(lp[0,:], lp[1,:])
    l1 = l1/l1[2]
    l2 = np.cross(lp[2,:], lp[3,:])
    l2 = l2/l2[2]
    l3 = np.cross(lp[4,:], lp[5,:])
    l3 = l3/l3[2]
    l4 = np.cross(lp[6,:], lp[7,:])
    l4 = l4/l4[2]
    
    p1_at_inf = np.cross(l1, l2)
    p1_at_inf = p1_at_inf/p1_at_inf[2]
    
    p2_at_inf = np.cross(l3,l4)
    p2_at_inf = p2_at_inf/p2_at_inf[2]
    
    ### Show the point at infinity
    plt.scatter(p1_at_inf[0], p1_at_inf[1], c='g', label = 'point 1 at inf')
    plt.scatter(p2_at_inf[0], p2_at_inf[1], c='g', label = 'point 2 at inf')
    x_values = [p1_at_inf[0]/p1_at_inf[2], p2_at_inf[0]/p2_at_inf[2]]
    y_values = [p1_at_inf[1]/p1_at_inf[2], p2_at_inf[1]/p2_at_inf[2]]
    plt.plot(x_values, y_values,'g', label = "line at  inf", linewidth=2) 
    
    plt.legend()
    "The line at infinity"
    l_at_inf = np.cross(p1_at_inf, p2_at_inf)
    l_at_inf = l_at_inf/l_at_inf[2]
    
    
    """ The rectification matrix """
    
    H = np.row_stack([np.array([1,0,0]),np.array([0,1,0]), l_at_inf])
    
    ### verify that H^(-1T) * l_at_inf =[0 0 1]^T
    
    Linf = np.dot(np.linalg.inv(H.T), l_at_inf)
    
    ### warping
    tform = ProjectiveTransform(H)
    immTr = warp(np.array(imm), tform.inverse, output_shape =(imm.size[1], imm.size[0]))
    imm_aff = crop_imm(immTr, imm.size)
    plt.figure(2)
    plt.imshow(imm_aff)
    plt.title('Affine Image')
    return imm_aff, H


def affineToMetric(aff_imm, imm):
    """ LAST QUESTION of Exercise 2: note that we obtained an affine rectification 
        and not a metric rectification.
        To obtain a metric rectification from the affine one we need to find 2 constraints
        These two constraints are given by two pairs of orthogonal lines L1, M1
        and L2, M2 in the affine transformation corresponding to two pairs of 
        orthogonal lines in real world. These two constraints are 
        enough  to compute the dual conic at infinity
    """
    
    plt.imshow(aff_imm)    
    plt.title(
        'Choose 8 points so as to obtain a pair of orthogonal lines: lines of the orthogonal pair should not be parallel')
    l1A = plt.ginput(8, timeout=60, show_clicks=True)
    
    #first line
    x_values = [l1A[0][0], l1A[1][0]]
    y_values = [l1A[0][1], l1A[1][1]]
    plt.plot(x_values, y_values,'r', label = "line 1", linewidth=2)
    
    
    #### second line
    x_values = [l1A[2][0], l1A[3][0]]
    y_values = [l1A[2][1], l1A[3][1]]
    plt.plot(x_values, y_values,'r', label = "line 2", linewidth=2)
    
    #### third line
    x_values = [l1A[4][0], l1A[5][0]]
    y_values = [l1A[4][1], l1A[5][1]]
    plt.plot(x_values, y_values,'b', label = "line 3", linewidth=2)
    
    
    #### fourth line
    x_values = [l1A[6][0], l1A[7][0]]
    y_values = [l1A[6][1], l1A[7][1]]
    plt.plot(x_values, y_values,'b', label = "line 4", linewidth=2)
    
    plt.legend()
    
    
    LM = np.array(l1A,dtype= float)
    LM = np.column_stack([LM, np.matlib.repmat(1, len(LM),1)])
    """ Now we have to solve:
        L1^T  [KK^T 0] M1 = 0
              [ 0^T 0]
       Let L1 =[l11, l12,L13]^T and M1 =[m11,m12,m13]^T,
       then the above equation reduces to
       (l11*m11, l11*m12+l12*m11, l12*m12)s = 0
       where s is the upper triangular part of KK^T
    """
    L1 = np.cross(LM[0,:], LM[1,:])
    L1 =L1/L1[2]
    M1 = np.cross(LM[2,:],LM[3,:])
    M1 = M1/M1[2]
    
    constraint1 = (L1[0]*M1[0], L1[0]*M1[1]+L1[1]*M1[0], L1[1]*M1[1])
    
    L2 = np.cross(LM[4,:], LM[5,:])
    L2 =L2/L2[2]
    M2 = np.cross(LM[6,:],LM[7,:])
    M2 = M2/M2[2]
    
    constraint2 = (L2[0]*M2[0], L2[0]*M2[1]+L2[1]*M2[0], L2[1]*M2[1])
    
    ### We now obtain s as the null space of vecs
    C = np.row_stack([constraint1,constraint2])
    
    s = scipy.linalg.null_space(C)
    s1 = np.insert(s,2,s[1])
    S = np.reshape(s1,(2,2))
    
    """ If S is singular you might have chosen wrong points"""
    ##verify S is non singular you might have chosen wrong points
    np.all(np.linalg.eigvals(S) > 0)
    ### Verify:
    cc1 = np.dot(L1[:2],np.dot(S,M1[:2]))
    cc2 = np.dot(L2[:2],np.dot(S,M2[:2]))
    np.allclose(cc1, 0.0, rtol=1e-05, atol=1e-08, equal_nan=False)
    np.allclose(cc2, 0.0, rtol=1e-05, atol=1e-08, equal_nan=False)
    
    ## To obtain K compute now the cholesky factorization of S
    K = linalg.cholesky(S, lower = False)
    Kx = K.copy()
    Kx[1,0] = K[0,1]
    ###############################
    
    HA = np.row_stack([np.column_stack([Kx,np.zeros(2)]), np.array([0,0,1])])

    ### Verify
    np.allclose(K@K.T-S, 0.0, rtol=1e-05, atol=1e-08, equal_nan=False)
    tform = AffineTransform(HA.T)
    immM = warp(np.array(aff_imm), tform,  output_shape = (2*imm.size[0],2*imm.size[1]))
    imm_metric = crop_imm(immM, imm.size)
    plt.figure(2)
    plt.imshow(imm_metric)
    
    return imm_metric


def main():
    plt.close('all')
    imm = Image.open('piastrelle.png')
    aff_imm,Hp = projToAffine(imm)
    metr_imm = affineToMetric(aff_imm, imm)
    plt.close('all')
    rcParams['figure.figsize'] = 18,11
    fig, axs = plt.subplots(1, 3)
    fig = plt.gcf()
    fig.suptitle("Transformation from projective to affine and from affine to metric", fontsize=14)
    axs[0].imshow(imm)
    axs[0].title.set_text('original image')
    axs[1].imshow(aff_imm)
    axs[1].title.set_text('affine transformation')
    axs[2].imshow(metr_imm)
    axs[2].title.set_text('metric transformation')
