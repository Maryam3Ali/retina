''' script to test clasification'''
#%%
import numpy as np
import napari
import matplotlib.pyplot as plt
import tifffile
import pandas as pd
import glasbey
# original glasbey
glas = glasbey.create_palette(256)

ffolder = r'G:\office\work\git\retina\DATA\2018004_1_1'
imageFile = '2018004_71_960nm_116fr_3Pro_1_1-Ch2-_intensity_image.tif'
tauFile = '2018004_1_1_tm_Ch2.xlsx'
classFile = '2018004_1_1_ch2_classification.ods'
labelFile = '2018004_1_1_ch2_labels.txt'

class GrainId:
    name = {'BACKGROUND': 0,
            'L': 1,
            'M': 2,
            'ML':3,
            'NON':4}

#%% data loading

image = tifffile.imread(ffolder +'/' + imageFile)
tau = pd.read_excel(ffolder +'/' + tauFile, sheet_name="TauMean 1").to_numpy()[::-1,:]

_sheet = pd.read_excel(ffolder +'/' + classFile, sheet_name="Sheet1")
gPos = np.vstack((_sheet['X'].to_numpy(),_sheet['Y'].to_numpy())).T.astype(int)


_gClass = _sheet['classification'].to_list()
gClass = np.array([GrainId.name[str.upper(ii)] for ii in _gClass])

givenLabel = pd.read_csv(ffolder +'/' + labelFile, sep='\t').to_numpy()


#%%

classImage = np.zeros_like(image)
classImage[gPos[:,-1],gPos[:,-2]] = gClass

#%%

viewer = napari.Viewer()

viewer.add_image(image)
viewer.add_labels(givenLabel, colormap=glas)
#viewer.add_image(tau)
#viewer.add_image(classImage)


#background = restoration.rolling_ball(image)
#cImage = image-background

#viewer.add_image(cImage)

from skimage import data, restoration, util
from skimage.filters import threshold_otsu
import skimage as ski
from skimage.filters import sobel

background = restoration.rolling_ball(image)
cImage = image-background
viewer.add_image(cImage)

#block_size = 15
#thresh = ski.filters.threshold_local(cImage, block_size, offset=0)
#viewer.add_image(thresh)
thresh = ski.filters.threshold_otsu(cImage)
#thresh = ski.filters.threshold_minimum(cImage)
#thresh = 6500
binary = image > thresh
viewer.add_image(binary)

from skimage.morphology import binary_erosion, binary_closing, binary_dilation


block_size = 35
local_thresh = ski.filters.threshold_local(image, block_size, offset=0)
binary_local = image > local_thresh


viewer.add_image(binary_local*binary)

#%%



edges = sobel(cImage)

viewer.add_image(edges)

local_max_coords = ski.feature.peak_local_max(
    cImage, min_distance=3, exclude_border=False
)
local_max_mask = np.zeros(image.shape, dtype=bool)
local_max_mask[tuple(local_max_coords.T)] = True
markers = ski.measure.label(local_max_mask)

viewer.add_image(markers)

cells = ski.segmentation.watershed(edges, markers, mask = binary_local*binary)

#viewer.add_image(cells)

viewer.add_labels(cells, colormap=glas)



#%%
from skimage.filters import threshold_multiotsu
thresholds = threshold_multiotsu(image)
regions = np.digitize(image, bins=thresholds)

#%%
viewer.add_image(regions)





# %% segmentation

from scipy import ndimage as ndi
import skimage as ski


distance = ndi.distance_transform_edt(image)

# minimal destance between granules
local_max_coords = ski.feature.peak_local_max(
    distance, min_distance=1, exclude_border=False
)
local_max_mask = np.zeros(distance.shape, dtype=bool)
local_max_mask[tuple(local_max_coords.T)] = True
markers = ski.measure.label(local_max_mask)

segmented_cells = ski.segmentation.watershed(-distance, markers, mask=image)

viewer.add_image(segmented_cells)

# %%
from skimage.morphology import binary_erosion, binary_closing, binary_dilation


block_size = 3
local_thresh = ski.filters.threshold_local(image, block_size, offset=10)
binary_local = image > local_thresh

newIm = binary_closing(binary_dilation(binary_erosion(binary_local,footprint=np.ones((5, 5))),
                                       footprint=np.ones((5, 5))))

viewer.add_image(newIm)


#%%




#viewer.add_image(segmented_cells)
# %%
from skimage.filters import sobel

edges = sobel(image)
# %%
viewer.add_image(edges)
# %%

#%%
from skimage import data, restoration, util

background = restoration.rolling_ball(image)
cImage = image-background

viewer.add_image(cImage)

from skimage.filters import sobel

edges = sobel(cImage)
viewer.add_image(edges)


# %%

markers = np.zeros_like(cImage)
markers[cImage < 20000] = 1
markers[cImage > 20000] = 2

fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(markers, cmap=plt.cm.nipy_spectral)
ax.set_title('markers')
ax.set_axis_off()

segmentation_coins = ski.segmentation.watershed(edges, markers)

fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(segmentation_coins, cmap=plt.cm.gray)
ax.set_title('segmentation')
ax.set_axis_off()
# %%

edgesBinary = edges>0.05

viewer.add_image(edgesBinary)
# %%
