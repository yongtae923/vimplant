# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 22:11:08 2021

@author: Lozano & van Hoof

"""

import numpy as np
from matplotlib import pyplot as plt

def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    
    """

    x = np.arange(0, size, 1, 'float32')
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

def sector_mask(shape,centre,radiusLow,radiusHigh,angle_range):

    """
    source original code (only sector):
    https://stackoverflow.com/questions/18352973/mask-a-circular-sector-in-a-numpy-array
    
    Return a boolean mask for a circular sector. The start/stop angles in  
    `angle_range` should be given in clockwise order.
    
    *Modification: I modified the code 
    so a LowAngle and a HighAngle can be defined for the sector
    
    """
    
    x,y = np.ogrid[:shape[0],:shape[1]]
    cx,cy = centre
    tmin,tmax = np.deg2rad(angle_range)
    
    # ensure stop angle > start angle
    if tmax < tmin:
            tmax += 2*np.pi
    
    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = np.arctan2(x-cx,y-cy) - tmin
    
    # wrap angles between 0 and 2*pi
    theta %= (2*np.pi)
    
    # circular mask
    circmaskLow = r2 >= radiusLow*radiusLow
    circmaskHigh = r2 <= radiusHigh*radiusHigh
    circmask = circmaskLow* circmaskHigh
    
    # angular mask
    anglemask = theta <= (tmax-tmin)
    
    return circmask*anglemask

#%% Complete gaussian
def complete_gauss(windowsize=1000, fwhm=400, radiusLow=0, radiusHigh=500, center=None, plotting=True):
    matrix = makeGaussian(windowsize, fwhm, center)

    ms = matrix.shape
    ms_2 = int(ms[0]/2)
    angle1 = -90
    angle2 = 90
    mask = sector_mask(matrix.shape,(ms_2,ms_2),radiusLow, radiusHigh,(angle1,angle2))
    matrix[~mask] = 0
    
    if plotting:
        plt.figure(dpi=180)
        plt.imshow(matrix, cmap = 'jet')
        plt.axis('off')
        plt.show()
    
    return matrix

#%% Outer ring
def outer_ring(windowsize=1000, fwhm=400, radiusLow=250, radiusHigh=500, center=None, plotting=True):
    matrix = makeGaussian(windowsize, fwhm, center)

    ms = matrix.shape
    ms_2 = int(ms[0]/2)
    angle1 = -90
    angle2 = 90
    mask = sector_mask(matrix.shape,(ms_2,ms_2),radiusLow, radiusHigh,(angle1,angle2))
    matrix[~mask] = 0
    if plotting:
        plt.figure(dpi=180)
        plt.imshow(matrix, cmap = 'jet')
        plt.axis('off')
        plt.show()
    
    return matrix

#%% Inner ring
def inner_ring(windowsize=1000, fwhm=400, radiusLow=0, radiusHigh=250, center=None, plotting=True):
    matrix = makeGaussian(windowsize, fwhm, center)
    ms = matrix.shape
    ms_2 = int(ms[0]/2)

    angle1 = -90
    angle2 = 90
    mask = sector_mask(matrix.shape,(ms_2,ms_2),radiusLow, radiusHigh,(angle1,angle2))
    matrix[~mask] = 0
    if plotting:
        plt.figure(dpi=180)
        plt.imshow(matrix, cmap = 'jet')
        plt.axis('off')
        plt.show()
    
    return matrix

#%% Upper sector 45 degrees
def upper_sector(windowsize=1000, fwhm=400, radiusLow=0, radiusHigh=500, center=None, angle1=-90, angle2=-45, plotting=True):
    matrix = makeGaussian(windowsize, fwhm, center)
    ms = matrix.shape
    ms_2 = int(ms[0]/2)
    mask = sector_mask(matrix.shape,(ms_2,ms_2),radiusLow, radiusHigh,(angle1,angle2))
    matrix[~mask] = 0
    if plotting:
        plt.figure(dpi=180)
        plt.imshow(matrix, cmap = 'jet')
        plt.axis('off')
        plt.show()
        
    return matrix

#%% Lower sector 45 degrees
def lower_sector(windowsize=1000, fwhm=400, radiusLow=0, radiusHigh=500, center=None, angle1=45, angle2=90, plotting=True):
    matrix = makeGaussian(windowsize, fwhm, center)

    ms = matrix.shape
    ms_2 = int(ms[0]/2)
    mask = sector_mask(matrix.shape,(ms_2,ms_2),radiusLow, radiusHigh,(angle1,angle2))
    matrix[~mask] = 0
    if plotting:
        plt.figure(dpi=180)
        plt.imshow(matrix, cmap = 'jet')
        plt.axis('off')
        plt.show()
    
    return matrix

#%% Square target
def square_target(windowsize=1000, fwhm=400, square_size=200, center=None, plotting=True):
    """
    Create a square-shaped target map.
    
    Parameters:
    - windowsize: Size of the output matrix
    - fwhm: Full Width at Half Maximum for gaussian smoothing
    - square_size: Size of the square (in pixels)
    - center: Center position of the square (if None, uses center of matrix)
    - plotting: Whether to plot the result
    """
    matrix = np.zeros((windowsize, windowsize), dtype='float32')
    
    # Set center position
    if center is None:
        center_x = windowsize // 2
        center_y = windowsize // 2
    else:
        center_x, center_y = center
    
    # Calculate square boundaries
    half_size = square_size // 2
    x_start = max(0, center_x - half_size)
    x_end = min(windowsize, center_x + half_size)
    y_start = max(0, center_y - half_size)
    y_end = min(windowsize, center_y + half_size)
    
    # Create square mask
    matrix[y_start:y_end, x_start:x_end] = 1.0
    
    # Apply gaussian smoothing if fwhm > 0
    if fwhm > 0:
        # Use scipy's built-in gaussian filter for better memory efficiency
        from scipy import ndimage
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # FWHM to sigma 변환
        
        try:
            matrix = ndimage.gaussian_filter(matrix, sigma=sigma, mode='constant', cval=0.0)
        except MemoryError:
            # If still memory error, use smaller sigma
            smaller_sigma = sigma / 2
            matrix = ndimage.gaussian_filter(matrix, sigma=smaller_sigma, mode='constant', cval=0.0)
    
    if plotting:
        plt.figure(dpi=180)
        plt.imshow(matrix, cmap='jet')
        plt.axis('off')
        plt.title(f'Square Target (size={square_size}, fwhm={fwhm})')
        plt.show()
    
    return matrix

#%% Equilateral triangle target
def triangle_target(windowsize=1000, fwhm=400, triangle_size=200, center=None, plotting=True):
    """
    Create an equilateral triangle-shaped target map.
    
    Parameters:
    - windowsize: Size of the output matrix
    - fwhm: Full Width at Half Maximum for gaussian smoothing
    - triangle_size: Size of the triangle (height in pixels)
    - center: Center position of the triangle (if None, uses center of matrix)
    - plotting: Whether to plot the result
    """
    matrix = np.zeros((windowsize, windowsize), dtype='float32')
    
    # Set center position
    if center is None:
        center_x = windowsize // 2
        center_y = windowsize // 2
    else:
        center_x, center_y = center
    
    # Calculate triangle parameters
    height = triangle_size
    # For equilateral triangle: side_length = height * 2 / sqrt(3)
    side_length = height * 2 / np.sqrt(3)
    half_side = side_length / 2
    
    # Triangle vertices (pointing upward)
    # Top vertex
    top_x = center_x
    top_y = center_y - height // 2
    
    # Bottom left vertex
    bottom_left_x = center_x - half_side
    bottom_left_y = center_y + height // 2
    
    # Bottom right vertex
    bottom_right_x = center_x + half_side
    bottom_right_y = center_y + height // 2
    
    # Create triangle mask using barycentric coordinates
    for y in range(windowsize):
        for x in range(windowsize):
            # Calculate barycentric coordinates
            denom = (bottom_left_y - bottom_right_y) * (top_x - bottom_right_x) + (bottom_right_x - bottom_left_x) * (top_y - bottom_right_y)
            
            if abs(denom) > 1e-10:  # Avoid division by zero
                a = ((bottom_left_y - bottom_right_y) * (x - bottom_right_x) + (bottom_right_x - bottom_left_x) * (y - bottom_right_y)) / denom
                b = ((bottom_right_y - top_y) * (x - bottom_right_x) + (top_x - bottom_right_x) * (y - bottom_right_y)) / denom
                c = 1 - a - b
                
                # Check if point is inside triangle
                if a >= 0 and b >= 0 and c >= 0:
                    matrix[y, x] = 1.0
    
    # Apply gaussian smoothing if fwhm > 0
    if fwhm > 0:
        # Use scipy's built-in gaussian filter for better memory efficiency
        from scipy import ndimage
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # FWHM to sigma 변환
        
        try:
            matrix = ndimage.gaussian_filter(matrix, sigma=sigma, mode='constant', cval=0.0)
        except MemoryError:
            # If still memory error, use smaller sigma
            smaller_sigma = sigma / 2
            matrix = ndimage.gaussian_filter(matrix, sigma=smaller_sigma, mode='constant', cval=0.0)
    
    if plotting:
        plt.figure(dpi=180)
        plt.imshow(matrix, cmap='jet')
        plt.axis('off')
        plt.title(f'Triangle Target (size={triangle_size}, fwhm={fwhm})')
        plt.show()
    
    return matrix