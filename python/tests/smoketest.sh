#!/bin/bash

# Script intended to "smoke test" the code.  This can be run in CI to help catch
# any fundamental incompatibilities that have snuck in with new releases of
# dependencies.  It does not do more in depth validation on the correctness of
# the results.

# Invocation with pre-defined PSF
deconvolution \
--data-path=flowdec/datasets/bars-25pct/data.tif \
--psf-path=flowdec/datasets/bars-25pct/kernel.tif \
--output-path=result.tif \
--n-iter=25 \
--log-level=DEBUG

# Invocation with dynamic PSF
echo '{"na": 0.75, "wavelength": 0.425, "size_z": 32, "size_x": 64, "size_y": 64}' > psf.json
deconvolution \
--data-path=flowdec/datasets/bars-25pct/data.tif \
--psf-config-path=psf.json \
--output-path=result.tif \
--n-iter=25 \
--log-level=DEBUG

