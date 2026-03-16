import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

orig = fits.getdata("debug_injection_outputs/original_projection.fits")
inj  = fits.getdata("debug_injection_outputs/injected_projection.fits")
mask = fits.getdata("debug_injection_outputs/detector_mask.fits")

y = 1516
x = 1196


r = 20

orig_crop = orig[y-r:y+r, x-r:x+r]
inj_crop  = inj[y-r:y+r, x-r:x+r]
diff_crop = inj_crop - orig_crop
mask_crop = mask[y-r:y+r, x-r:x+r]

fig, ax = plt.subplots(1,4, figsize=(14,4))

ax[0].imshow(orig_crop, cmap="gray")
ax[0].set_title("original")

ax[1].imshow(inj_crop, cmap="gray")
ax[1].set_title("injected")

ax[2].imshow(diff_crop, cmap="inferno")
ax[2].set_title("difference")

ax[3].imshow(mask_crop, cmap="gray")
ax[3].set_title("detector mask")

for a in ax:
    a.axis("off")

plt.tight_layout()
plt.show()