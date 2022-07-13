import numpy as np
import skimage.transform
import skimage.filters
import logging

class DecalsAugmentations():
    def __init__(self, augmentations='jc', params={}):

        self.augmentations_str = augmentations
        self.npix_out = params.get('npix_out', 96)
        self.jitter_lim = params.get('jitter_lim', 0)
        self.only_dered = params.get('only_dered', False)
        self.only_red = params.get('only_dered', False)
        self.ebv_max = params.get('ebv_max', 1.0)
        self.gn_uniform = params.get('gn_uniform', False)
        self.gb_uniform = params.get('gb_uniform', False)
        self.gr_uniform = params.get('gr_uniform', False)
        self.verbose = params.get('verbose', False)

        # Dictionary to convert between abbreviation and full augmentation string
        self.aug_to_name = {
            'gr': 'GalacticReddening',
            'rr': 'RandomRotate',
            'ss': 'SizeScale',
            'gb': 'GaussianBlur',
            'jc': 'JitterCrop',
            'cc': 'CenterCrop',
            'gn': 'GaussianNoise',
            'rg': 'ToRGB',
        }
        
        self.aug_to_func = {
            'gr': self.add_Reddening,
            'rr': self.add_RandomRotate,
            'ss': self.add_SizeScale,
            'gb': self.add_GaussianBlur,
            'jc': self.add_JitterCrop,
            'cc': self.add_CenterCrop,
            'gn': self.add_GaussianNoise,
            'rg': self.add_ToRGB,
        }   
    
    # Augmentation functions are listed in the order that they (mostly) should be called
    def add_JitterCrop(self):
        self.augmentations_names.append(self.aug_to_name['jc'])    
        self.augmentations.append(JitterCrop(outdim=self.npix_out, jitter_lim=self.jitter_lim))

    def add_CenterCrop(self):
        self.augmentations_names.append(self.aug_to_name['cc'])    
        self.augmentations.append(JitterCrop(outdim=self.npix_out, jitter_lim=0))

    def add_RandomRotate(self):
        self.augmentations_names.append(self.aug_to_name['rr'])    
        self.augmentations.append(RandomRotate())
 
    def add_Reddening(self):
        self.augmentations_names.append(self.aug_to_name['gr'])
        self.augmentations.append(Reddening(
            only_dered = self.only_dered,
            only_red = self.only_red,
            ebv_max = self.ebv_max,
            uniform = self.gr_uniform,
        ))
        
    def add_SizeScale(self):
        self.augmentations_names.append(self.aug_to_name['ss'])    
        self.augmentations.append(SizeScale())
        
    def add_GaussianNoise(self):
        self.augmentations_names.append(self.aug_to_name['gn'])    
        self.augmentations.append(GaussianNoise(uniform=self.gn_uniform))

    def add_GaussianBlur(self):
        self.augmentations_names.append(self.aug_to_name['gb'])    
        self.augmentations.append(GaussianBlur(uniform=self.gb_uniform))
        
    def add_ToRGB(self):
        self.augmentations_names.append(self.aug_to_name['rg'])    
        self.augmentations.append(ToRGB())
             
    def add_augmentations(self, augs_manual: str=None):
        self.augmentations = []
        self.augmentations_names = []
        
        # split every two characters
        if augs_manual is None:
            augmentations_use = self.augmentations_str
        if augs_manual is not None:
            augmentations_use = augs_manual
        
        augs = [augmentations_use[i:i+2] for i in range(0, len(augmentations_use), 2)]

        for aug in augs:
            if aug not in self.aug_to_func.keys():
                sys.exit(f"Augmentation abbreviation {aug} is not an available key. Choose from", self.aug_to_name.key())
                
            self.aug_to_func[aug]() # () required to actually call function

        if self.verbose:
            logging.info(f"Using augmentations: {self.augmentations_names}")

        return self.augmentations, self.augmentations_names
    
class RandomRotate:
    '''takes in image of size (npix, npix, nchannel), flips l/r and or u/d, then rotates (0,360)'''
    def __call__(self, image):    
        
        if np.random.randint(0, 2)==1:
            image = np.flip(image, axis=0)
        if np.random.randint(0, 2)==1:
            image = np.flip(image, axis=1)

        return skimage.transform.rotate(image, float(360*np.random.rand(1)))

class JitterCrop:
    '''takes in image of size (npix, npix, nchannel), 
    jitters by uniformly drawn (-jitter_lim, jitter_lim),
    and returns (outdim, outdim, nchannel) central pixels'''

    def __init__(self, outdim=96, jitter_lim=7):
        self.outdim = outdim
        self.jitter_lim = jitter_lim
        
    def __call__(self, image):                            
        if self.jitter_lim:
            center_x = image.shape[0]//2 + int(np.random.randint(-self.jitter_lim, self.jitter_lim+1, 1))
            center_y = image.shape[0]//2 + int(np.random.randint(-self.jitter_lim, self.jitter_lim+1, 1))
        else:
            center_x = image.shape[0]//2
            center_y = image.shape[0]//2
        offset = self.outdim//2

        return image[(center_x-offset):(center_x+offset), (center_y-offset):(center_y+offset)]

class SizeScale:
    '''takes in image of size (npix, npix, nchannel), and scales the size larger or smaller
    anti-aliasing should probably be enabled when down-sizing images to avoid aliasing artifacts
    
    This augmentation changes the number of pixels in an image. After sizescale, we still need enough
    pixels to allow for jitter crop to not run out of bounds. Therefore, 
    
    scale_min >= (outdim + 2*jitter_lim)/indim
    
    if outdim = 96, and indim=152 and jitter_lim = 7, then scale_min >= 0.73.
    
    When using sizescale, there is a small possibility that one corner of the image can be set to 0 in randomrotate,
    then the image can be scaled smaller, and if the image is jittered by near the maximum allowed value, that these
    0s will remain in a corner of the final image. Adding Gaussian noise after all the other augmentations should 
    remove any negative effects of this small 0 patch.
    '''
    
    def __init__(self, scale_min=0.9, scale_max=1.1):
        
        if scale_min < 0.73:
            logging.info('scale_min reset to minimum value, given 152 pix image and 96 pix out')
            scale_min = 0.73
        self.scale_min = scale_min
        self.scale_max = scale_max
        
    def __call__(self, image):    

        scalei = np.random.uniform(self.scale_min, self.scale_max)
        
        return skimage.transform.rescale(image, scalei, anti_aliasing=False, channel_axis=True)

    
class GaussianNoise:
    '''adds Gaussian noise consistent from distribution fit to decals south \sigma_{pix,coadd}
    (see https://www.legacysurvey.org/dr9/nea/) as measured from 43e6 samples with zmag<20.

    Images already have noise level when observed on sky, so we do not want 
    to add a total amount of noise, we only want to augment images by the 
    difference in noise levels between various objects in the survey.
    
    1/sigma_pix^2 = psf_depth * [4pi (psf_size/2.3548/pixelsize)^2],
    where psf_size from the sweep catalogue is fwhm in arcsec, pixelsize=0.262 arcsec, 
    and 2.3548=2*sqrt(2ln(2)) converst from fwhm to Gaussian sigma
    
    noise in each channel is uncorrelated, as images taken at different times/telescopes.

    A lognormal fit matches the measured noise distribution better than Gaussian. Fit with scipy,
    which has a different paramaterization of the log normal than numpy.random 
    (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html):
    
    shape, loc, scale -> np.random.lognormal(np.log(scale), shape, size=1) + loc

    # g: (min, max)=(0.001094, 0.013), Lognormal fit (shape, loc, scale)=(0.2264926, -0.0006735, 0.0037602)
    # r: (min, max)=(0.001094, 0.018), Lognormal fit (shape, loc, scale)=(0.2431146, -0.0023663, 0.0067417)
    # z: (min, max)=(0.001094, 0.061), Lognormal fit (shape, loc, scale)=(0.1334844, -0.0143416, 0.0260779)
    '''

    def __init__(self, scaling = [1.], mean=0, im_dim=96, im_ch=3, decals=True, uniform=False):
        self.mean = mean
        self.decals = decals
        self.im_ch = im_ch
        self.im_dim = im_dim
        self.uniform = uniform

        # Log normal fit paramaters
        self.shape_dist = np.array([0.2264926, 0.2431146, 0.1334844])
        self.loc_dist  = np.array([-0.0006735, -0.0023663, -0.0143416])
        self.scale_dist  = np.array([0.0037602, 0.0067417, 0.0260779])
        
        self.sigma_dist  = np.log(self.scale_dist)
    
        # noise in channels is uncorrelated, as images taken at dirrerent times/telescopes
        self.noise_ch_min = np.array([0.001094, 0.001094, 0.001094])
        self.noise_ch_max = np.array([0.013, 0.018, 0.061])

    def __call__(self, image):
        
        # draw 'true' noise level of each channel from lognormal fits
        self.sigma_true = np.random.lognormal(self.sigma_dist, self.shape_dist) + self.loc_dist
    
        if self.uniform:
            # draw desired augmented noise level from uniform, to target tails more
            self.sigma_final = np.random.uniform(self.noise_ch_min, self.noise_ch_max)
        else:
            self.sigma_final = np.random.lognormal(self.sigma_dist, self.shape_dist) + self.loc_dist

        # Gaussian noise adds as c^2 = a^2 + b^2
        self.sigma_augment = self.sigma_final**2 - self.sigma_true**2
        self.sigma_augment[self.sigma_augment < 0.] = 0.
        self.sigma_augment = np.sqrt(self.sigma_augment)
        
        for i in range(self.im_ch):
            if self.sigma_augment[i] > 0.:
                image[:,:,i] += np.random.normal(self.mean, self.sigma_augment[i], size = (self.im_dim, self.im_dim))

        return image

class GaussianBlur:
    '''adds Gaussian PSF blur consistent from distribution fit to decals psf_size
    from sweep catalogues as measured from 2e6 spectroscopic samples. 
    Images have already been filtered by PSF when observed on sky, so we do not want 
    to smooth images using a total smoothing, we only want to augment images by the 
    difference in smoothings between various objects in the survey.
    
    sigma = psf_size / pixelsize / 2.3548,
    where psf_size from the sweep catalogue is fwhm in arcsec, pixelsize=0.262 arcsec, 
    and 2.3548=2*sqrt(2ln(2)) converst from fwhm to Gaussian sigma
    
    PSF in each channel is uncorrelated, as images taken at different times/telescopes.

    A lognormal fit matches the measured PSF distribution better than Gaussian. Fit with scipy,
    which has a different paramaterization of the log normal than numpy.random 
    (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html):
    
    shape, loc, scale -> np.random.lognormal(np.log(scale), shape, size=1) + loc

    # g: (min, max)=(1.3233109, 5), Lognormal fit (shape, loc, scale)=(0.2109966, 1.0807153, 1.3153171)
    # r: (min, max)=(1.2667341, 4.5), Lognormal fit (shape, loc, scale)=(0.3008485, 1.2394326, 0.9164757)
    # z: (min, max)=(1.2126263, 4.25), Lognormal fit (shape, loc, scale)=(0.3471172, 1.1928363, 0.8233702)
    '''

    def __init__(self, scaling = [1.], im_dim=96, im_ch=3, decals=True, uniform=False):
        self.decals = decals
        self.im_ch = im_ch
        self.im_dim = im_dim
        self.uniform = uniform
        
        # Log normal fit paramaters
        self.shape_dist = np.array([0.2109966, 0.3008485, 0.3471172])
        self.loc_dist  = np.array([1.0807153, 1.2394326, 1.1928363])
        self.scale_dist  = np.array([1.3153171, 0.9164757, 0.8233702])
        
        self.sigma_dist  = np.log(self.scale_dist)

        self.psf_ch_min = np.array([1.3233109, 1.2667341, 1.2126263])
        self.psf_ch_max = np.array([5., 4.5, 4.25])

    def __call__(self, image):
        # noise in channels is uncorrelated, as images taken at different times/telescopes

        # draw 'true' noise level of each channel from lognormal fits
        self.sigma_true = np.random.lognormal(self.sigma_dist, self.shape_dist) + self.loc_dist
    
        if self.uniform:
            # draw desired augmented noise level from uniform, to target tails more
            self.sigma_final = np.random.uniform(self.psf_ch_min, self.psf_ch_max)
        else:
            self.sigma_final = np.random.lognormal(self.sigma_dist, self.shape_dist) + self.loc_dist

        # Gaussian noise adds as c^2 = a^2 + b^2
        self.sigma_augment = self.sigma_final**2 - self.sigma_true**2
        self.sigma_augment[self.sigma_augment < 0.] = 0.
        self.sigma_augment = np.sqrt(self.sigma_augment)
        
#         logging.info(self.sigma_true, self.sigma_final, self.sigma_augment)
        for i in range(self.im_ch):
            if self.sigma_augment[i] > 0.:
                image[:,:,i] = skimage.filters.gaussian(image[:,:,i], sigma=self.sigma_augment[i], mode='reflect')

        return image

class Reddening:
    '''De-reddens image, and can re-redden with random sampling of ebv.
    
    A lognormal fit matches the measured E(B-V) distribution better than Gaussian. Fit with scipy,
    which has a different paramaterization of the log normal than numpy.random 
    (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html):
    
    shape, loc, scale -> np.random.lognormal(np.log(scale), shape, size=1) + loc

    # (min, max)=(0.00, 1.0), Lognormal fit (shape, loc, scale)=(0.67306, 0.001146, 0.03338)
    '''
  

    def __init__(self, only_dered=False, only_red=False, uniform=False, filters_use = ['g', 'r', 'z'], ebv_max=1.0):
        self.only_dered = only_dered
        self.only_red = only_red
        self.uniform = uniform
        
        self.filters_use = filters_use

        # Log normal fit paramaters
        self.shape_dist = 0.67306
        self.loc_dist = 0.001146
        self.scale_dist = 0.03338
        
        self.sigma_dist  = np.log(self.scale_dist)

        self.ebv_min = 0.00
        self.ebv_max = ebv_max
        
    def ebv_to_transmission(self, ebv):
        # ebv to transmission is just a simple power law I fit for each band - works perfectly
        # (I couldnt figure it out from reference paper https://ui.adsabs.harvard.edu/abs/1998ApJ...500..525S/abstract)

        # ebv = Galactic extinction E(B-V) reddening from SFD98, used to compute MW_TRANSMISSION
        # transmission = Galactic transmission in g filter in linear units [0,1]
        filters   = ['g', 'r', 'z', 'W1', 'W2']
        exponents = np.array([-1.2856, -0.866, -0.4844, -0.0736, -0.04520])

        nfilters     = len(filters)
        nfilters_use = len(self.filters_use)

        exponents = {filters[i]: exponents[i] for i in range(nfilters)}
        exponents_use  = np.array([exponents[fi] for fi in self.filters_use])

        transmission_ebv = 10**(ebv*exponents_use)

        return transmission_ebv

    def __call__(self, list_or_image):
        
        if type(list_or_image)==list: # if list then deredden the image    
            raw_image = list_or_image[0]
            ebv_i = list_or_image[1]
            
        else:
            # don't have ebv. Assume random value
            raw_image = list_or_image
    
            ebv_i = np.random.lognormal(self.sigma_dist, self.shape_dist) + self.loc_dist
        
        old_transmission = self.ebv_to_transmission(ebv_i)
                                     
        if self.only_dered:
            return raw_image/old_transmission
            
        if self.uniform:
            new_ebv = np.random.uniform(0, self.ebv_max)
        else:
            new_ebv = np.random.lognormal(self.sigma_dist, self.shape_dist) + self.loc_dist
            
        new_transmission = self.ebv_to_transmission(new_ebv)
        
        if self.only_red:
            return np.float32(raw_image * new_transmission)
        else:
            return np.float32(raw_image * new_transmission/old_transmission)


class ToRGB:
    '''takes in image of size (npix, npix, nchannel), 
    and converts from native telescope image scaling to rgb'''

    def __init__(self, scales=None, m=0.03, Q=20, bands=['g', 'r', 'z']):
        rgb_scales = {'u': (2,1.5),
                      'g': (2,6.0),
                      'r': (1,3.4),
                      'i': (0,1.0),
                      'z': (0,2.2)}
        if scales is not None:
            rgb_scales.update(scales)
            
        self.rgb_scales = rgb_scales
        self.m = m
        self.Q = Q
        self.bands = bands
        self.axes, self.scales = zip(*[rgb_scales[bands[i]] for i in range(len(bands))])
        
        # rearange scales to correspond to image channels after swapping
        self.scales = [self.scales[i] for i in self.axes]

    def __call__(self, image):
        
        image[:, :, np.arange(len(self.axes))] = image[:, :, self.axes]
        
        I = np.sum(np.maximum(0, image * self.scales + self.m), axis=-1)/len(self.bands)
        
        fI = np.arcsinh(self.Q * I) / np.sqrt(self.Q)
        I += (I == 0.) * 1e-6

        image = image * (self.scales + self.m * (fI / I)[..., np.newaxis]).astype(np.float32)

        image = np.clip(image, 0, 1)

        return image
    
    
