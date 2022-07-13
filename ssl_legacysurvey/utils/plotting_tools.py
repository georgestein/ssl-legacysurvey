import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.stats import binned_statistic_2d

from . import to_rgb

import math
import os



def plot_skymap(ra, dec, carr=None, field_str='', title=None, cmap='viridis', ptsize=20, pt_alpha=0.8, ra_shift=80., dra_lab=20):
    """Scatter plot of (ra, dec, color) in aitoff projection
    
    Parameters                                                                                  
    ----------                                                                                  
    ra : array                                                                        
        right ascension in deg                                                               
    dec : array                                                                        
        declination in deg
    carr : array                                                                        
        array to color data points by
    field_str : str
        colorbar label, if carr provided
    """
    ra_shift_rad = np.deg2rad(ra_shift)
    cm = plt.get_cmap(cmap)

    plt.figure(figsize=(16,8))
    plt.subplot(111, projection='aitoff')
    plt.grid(True)
    
    # default rotates to same frame as figures
    ra = np.deg2rad(ra)
    ra[ra>np.pi] -= 2*np.pi
    ra += ra_shift_rad
    ra = (ra+np.pi)%(2*np.pi)-np.pi 
    ra *= -1 # flip so ra increases to left (silly astronomers)

    dec = np.deg2rad(dec)
    
    plt.scatter(ra, dec, c=carr, label=field_str, s=ptsize, cmap=cm, alpha=pt_alpha)

    # set figure aesthetics
    plt.plot(np.linspace(-np.pi, np.pi, 360), np.deg2rad(np.full((360,), 32.375)), c='grey', lw=1, ls=':') 

    if carr is not None:
        cb = plt.colorbar(label=field_str, fraction=0.02, pad=0.02)
        
    xticks = np.arange(-180+dra_lab, 180+dra_lab, dra_lab)
    xtick_labels = np.arange(0, 360, dra_lab).astype(int)[::-1]
    xtick_labels = np.roll(xtick_labels, int(xtick_labels.shape[0]//2-ra_shift//dra_lab )) 
    xtick_labels = ['{:d}$^o$'.format(i) for i in xtick_labels]
    plt.xticks(ticks=np.radians(xticks), labels=xtick_labels, rotation=30)
    plt.grid(True, linestyle=':')
    
    plt.title(title)
    plt.show()


def show_galaxies(images, ra, dec, display_radec=True,
                  title=None, is_previous_lens=None, is_new_lens=None, label_lenses=False,
                  nx=8, npix_show=96, nplt=None, savepath=None,
                  panel_size=[3,3], lw_rect=3, lw_border=1, ls_border='-', fontsize=12, colors=['C0', 'red']):
    """Plot images in an nx by len(images)//nx array
    
    Parameters
    ----------
    images : array
        an (N_img, npix_x, npix_y) array
    is_previous_lens: boolean array
        (N_img) array denoting whether the galaxy is a lens previously identified in literature
    is_new_lens: boolean array
        (N_img) array denoting whether the galaxy is a lens identified in out work
    label_lens: boolean
        whether or not to use lens labels
    nx: int
        number of images to display in each row
    nplt : int
        total number to plot if < N_img desired
    """
    
    nimg = images.shape[0]
    npix = images.shape[-1]
    
    ipix_start = npix//2 - npix_show//2
    ipix_end = npix//2 + npix_show//2

    ny = math.ceil(nimg/nx)
    if nplt is not None:
        ny = math.ceil(nplt/nx)
    nplt = nx*ny

    # instead of subplots plot as one large image - this is much faster
    image_full = np.ones((ny*npix_show, nx*npix_show, 3), dtype=np.float32)

    fig, ax = plt.subplots(1, 1, figsize=(nx*panel_size[0], ny*panel_size[1]))
    plt.subplots_adjust(wspace=0.00, hspace=0.00)

  
    fi = 0
    ntot = 0
    for i in range(ny):
        for j in range(nx):
            im = images[fi]

            image_full[i*npix_show+lw_border:(i+1)*npix_show-lw_border, 
                       j*npix_show+lw_border:(j+1)*npix_show-lw_border] = to_rgb.dr2_rgb(images[fi, :, 
                                                                                         ipix_start+lw_border:ipix_end-lw_border, 
                                                                                         ipix_start+lw_border:ipix_end-lw_border], 
                                                                                  ['g','r','z'])[::-1]

            fi+=1

            if fi >= nimg:
                break
        if fi >= nimg:
            break

    ax.imshow(image_full, interpolation='none')
    ax.axis('off')

    if label_lenses or display_radec:
        # color previous lenses
        fi = 0
        for i in range(ny):
            for j in range(nx):
                if label_lenses:
                    is_lens = is_previous_lens[fi] | is_new_lens[fi]

                    if is_lens:
                        ilp = is_previous_lens[fi]
                        if ilp:
                            ci = colors[0]
                        else:
                            ci = colors[1]


                        rect = patches.Rectangle((lw_border+j*npix_show,  lw_border+i*npix_show),
                                                 npix_show-lw_border*2-1, npix_show-lw_border*2-1, 
                                                 linewidth=lw_rect, ls=ls_border, edgecolor=ci, facecolor='none')#, alpha=0.2)#, 'none')
                        ax.add_patch(rect)

                if display_radec:
                    txt = '{:.4f} {:.4f}'.format(ra[fi], dec[fi])
                    ax.text(5+j*npix_show, (i-1)*npix_show + npix_show+5, txt, color='w', ha='left', va='top', fontsize=fontsize)
                
                fi+=1

                if fi >= nimg:
                    break
            if fi >= nimg:
                break

    if label_lenses:
        lw_leg = 2

        if ny > 1:
            ax.plot(0, 0, lw=lw_leg, color=colors[0], label="Previous Lens")
            ax.plot(0, 0, lw=lw_leg, color=colors[1], label="New Lens")
     
            ax.legend(ncol=2, bbox_to_anchor=(0.5, 1.01), loc='center', frameon=False)

    if title is not None:
        ax.set_title(title)

def scatter_plot_as_images(DDL, z_emb, inds_use, nx=8, ny=8, npix_show=96, iseed=13579, display_image=True):
    """Sample points from scatter plot and display as their original galaxy image
        
    Parameters
    ----------
    DDL : class instance
        DecalsDataLoader class instance
    z_emb: array
        (N, 2) array of the galaxies location in some compressed space. 
        If second axis has a dimensionality greater than 2 we only consider the leading two components.
    """
    z_emb = z_emb[:, :2] # keep only first two dimensions

    nplt = nx*ny

    img_full = np.zeros((ny*npix_show, nx*npix_show, 3)) + 255#, dtype=np.uint8) + 255

    xmin = z_emb[:,0].min()
    xmax = z_emb[:,0].max()
    ymin = z_emb[:,1].min()
    ymax = z_emb[:,1].max()

    dz_emb = 0.25
    dx_cent = z_emb[:,0].mean()
    dy_cent = z_emb[:,1].mean()

    dx_cent = 10.0
    dy_cent = 7.0

    # xmin = dx_cent - dz_emb
    # xmax = dx_cent + dz_emb
    # ymin = dy_cent - dz_emb
    # ymax = dy_cent + dz_emb

    binx = np.linspace(xmin,xmax, nx+1)
    biny = np.linspace(ymin,ymax, ny+1)

    ret = binned_statistic_2d(z_emb[:,0], z_emb[:,1], z_emb[:,1], 'count', bins=[binx, biny], expand_binnumbers=True)
    z_emb_bins = ret.binnumber.T

    inds_used = []
    inds_lin = np.arange(z_emb.shape[0])

    # First get all indexes that will be used
    for ix in range(nx):
        for iy in range(ny):
            dm = (z_emb_bins[:,0]==ix) & (z_emb_bins[:,1]==iy)
            inds = inds_lin[dm]

            np.random.seed(ix*nx+iy+iseed)
            if len(inds) > 0:
                ind_plt = np.random.choice(inds)
                inds_used.append(inds_use[ind_plt])


    # load in all images
    iimg = 0
    images = DDL.get_data(inds_used, fields='images', npix_out=npix_show)
    images = images['images']

    # Add each image as postage stamp in desired region  
    for ix in range(nx):
        for iy in range(ny):
            dm = (z_emb_bins[:,0] == ix) & (z_emb_bins[:,1]==iy)
            inds = inds_lin[dm]

            np.random.seed(ix*nx+iy+iseed)
            if len(inds) > 0:

                imi = to_rgb.dr2_rgb(images[iimg],
                                     ['g','r','z'])[::-1]

                img_full[iy*npix_show:(iy+1)*npix_show, ix*npix_show:(ix+1)*npix_show] = imi

                iimg += 1
                
    if display_image:
        plt.figure(figsize=(nx, ny))
        plt.imshow(img_full, origin='lower')#, interpolation='none')
        plt.axis('off')
        
    return img_full

def list_files(startpath):
    """Print directory structure and files within

    Taken from https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python
    """
    print(f'{startpath} structure and files:')

    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))
 
