"""Utility functions and constants necessary for Flowdec <--> PSFGenerator validation and comparison"""

# Configuration example taken from http://bigwww.epfl.ch/algorithms/psfgenerator/
DEFAULT_PSF_CONFIG = """
PSF-shortname=GL
ResLateral=100.0
ResAxial=250.0
NY=256
NX=256
NZ=65
Type=32-bits
NA=1.4
LUT=Fire
Lambda=610.0
Scale=Linear
psf-BW-NI=1.5
psf-BW-accuracy=Good
psf-RW-NI=1.5
psf-RW-accuracy=Good
psf-GL-NI=1.5
psf-GL-NS=1.33
psf-GL-accuracy=Good
psf-GL-ZPos=2000.0
psf-GL-TI=150.0
psf-TV-NI=1.5
psf-TV-ZPos=2000.0
psf-TV-TI=150.0
psf-TV-NS=1.0
psf-Circular-Pupil-defocus=100.0
psf-Circular-Pupil-axial=Linear
psf-Circular-Pupil-focus=0.0
psf-Oriented-Gaussian-axial=Linear
psf-Oriented-Gaussian-focus=0.0
psf-Oriented-Gaussian-defocus=100.0
psf-Astigmatism-focus=0.0
psf-Astigmatism-axial=Linear
psf-Astigmatism-defocus=100.0
psf-Defocus-DBot=30.0
psf-Defocus-ZI=2000.0
psf-Defocus-DTop=30.0
psf-Defocus-DMid=1.0
psf-Defocus-K=275.0
psf-Cardinale-Sine-axial=Linear
psf-Cardinale-Sine-defocus=100.0
psf-Cardinale-Sine-focus=0.0
psf-Lorentz-axial=Linear
psf-Lorentz-focus=0.0
psf-Lorentz-defocus=100.0
psf-Koehler-dMid=3.0
psf-Koehler-dTop=1.5
psf-Koehler-n1=1.0
psf-Koehler-n0=1.5
psf-Koehler-dBot=6.0
psf-Double-Helix-defocus=100.0
psf-Double-Helix-axial=Linear
psf-Double-Helix-focus=0.0
psf-Gaussian-axial=Linear
psf-Gaussian-focus=0.0
psf-Gaussian-defocus=100.0
psf-Cosine-axial=Linear
psf-Cosine-focus=0.0
psf-Cosine-defocus=100.0
psf-VRIGL-NI=1.5
psf-VRIGL-accuracy=Good
psf-VRIGL-NS2=1.4
psf-VRIGL-NS1=1.33
psf-VRIGL-TG=170.0
psf-VRIGL-NG=1.5
psf-VRIGL-TI=150.0
psf-VRIGL-RIvary=Linear
psf-VRIGL-ZPos=2000.0
"""

PSFGEN_PARAM_MAP = {
    'GL': {
        'ni0': 'psf-GL-NI', # Refractive index, immersion
        'ns': 'psf-GL-NS', # Specimen refractive index
        'pz': 'psf-GL-ZPos', # Particle distance from coverslip (nm for PSFGenerator, microns for Flowdec)
        'ti0': 'psf-GL-TI', # Working distance (microns for both)
        'wavelength': 'Lambda', # Emission wavelength (nm for PSFGenerator, micros for Flowdec)
        'res_lateral': 'ResLateral', # Lateral resolution (nm for PSFGenerator, microns for Flowdec)
        'res_axial': 'ResAxial', # Axial resolution (nm for PSFGenerator, microns for Flowdec)
        'na': 'NA', # Numerical aperture
        'size_x': 'NX', # Size X
        'size_y': 'NY', # Size Y
        'size_z': 'NZ'  # Size Z
    },
    'BW': {
        'ni0': 'psf-BW-NI', # Refractive index, immersion
        'wavelength': 'Lambda', # Emission wavelength
        'res_lateral': 'ResLateral', # Lateral resolution (nm for PSFGenerator, microns for Flowdec)
        'res_axial': 'ResAxial', # Axial resolution (nm for PSFGenerator, microns for Flowdec)
        'na': 'NA', # Numerical aperture
        'size_x': 'NX', # Size X
        'size_y': 'NY', # Size Y
        'size_z': 'NZ'  # Size Z
    }
}

DIVISORS = {
    'ResLateral': .001,
    'ResAxial': .001,
    'Lambda': .001,
    'Lambda': .001,
    'psf-GL-ZPos': .001
}

BW_PARAM_MAP = {
    'ni0': 'psf-BW-NI' # Refractive index, immersion
}

def get_default_psfgenerator_config():
    return psfgenerator_config_from_string(DEFAULT_PSF_CONFIG)

def psfgenerator_config_to_string(config):
    return '\n'.join(['{}={}'.format(k, v) for k, v in config.items()])

def psfgenerator_config_from_string(config):
    return dict([l.split('=') for l in config.split('\n') if l])
    
    
def flowdec_config_to_psfgenerator_config(config, mode='GL', accuracy='Best', dtype='32-bits'):
    if mode not in PSFGEN_PARAM_MAP:
        raise ValueError('Mode must be one of {} (given = {})'.format(list(PSFGEN_PARAM_MAP.keys()), mode))
    if accuracy not in ['Good', 'Better', 'Best']:
        raise ValueError('Accuracy level must be one of {} (given = {})'.format(['Good', 'Better', 'Best'], accuracy))
    if dtype not in ['32-bits', '64-bits']:
        raise ValueError('Data type must be one of {} (given = {})'.format(['32-bits', '64-bits'], dtype))
        
    res = get_default_psfgenerator_config()
    for k, v in config.items():
        
        # Ignore any parameters specific to Flowdec PSF generation alone
        if k not in PSFGEN_PARAM_MAP[mode]:
            continue
        
        # Get PSFGenerator config property name for flowdec property name 
        kt = PSFGEN_PARAM_MAP[mode][k]
        
        # Apply units transformation if necessary
        vt = v
        if kt in DIVISORS:
            vt = v / DIVISORS[kt]
        res[kt] = vt
    
    # Set the "short name" of the PSF type to calculate (BW=Born & Wolf, GL=Gibson & Lanni) as
    # well as the desired accuracy level
    res['PSF-shortname'] = mode
    res['psf-' + mode + '-accuracy'] = accuracy
    res['Type'] = dtype
    
    return res
    


