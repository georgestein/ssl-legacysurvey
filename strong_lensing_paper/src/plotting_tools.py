import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import src.to_rgb as to_rgb

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
 
