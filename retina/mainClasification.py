''' script to classify granules in the flim images '''
#%% import
import numpy as np
import napari
import matplotlib.pyplot as plt

from classificationFunction import *

#%%  set parameters

ffolder = r'G:\office\work\git\retina\DATA\2018004_1_1'
imageFile = '2018004_71_960nm_116fr_3Pro_1_1-Ch2-_intensity_image.tif'
tauFile = '2018004_1_1_tm_Ch2.xlsx'
classFile = '2018004_1_1_ch2_classification.ods'
columnName = 'classification'
#classFile = '2018004_1_1_ch2_classification2.xlsx'
#columnName = 'classification_Hala'

#labelFile = '2018004_1_1_ch2_labels.txt'


#%% process data

image, tau = loadData(ffolder, imageFile,tauFile)
classImage = loadClassification(ffolder,classFile,imageSize= image.shape, columnName= columnName)
intTauImage = getTauIntensityImage(image,tau)

# show original data
# original glasbey
#glas = glasbey.create_palette(256)
viewer = napari.Viewer()
viewer.add_image(image, name='intensity')
viewer.add_image(intTauImage, rgb=True)
viewer.add_labels(classImage,colormap=GrainId.colorNapari)

#%% show 2D histogram tau x intensity

H, xEdge, yEdge, binNumber = getTauIntensityHistogram(image,tau)

fig3, ax3 = plt.subplots()
ax3.pcolormesh(xEdge, yEdge, H, cmap='rainbow')
ax3.set_title('Tau Intensity pixels')
ax3.set_ylabel('intensity /a.u.')
ax3.set_xlabel('tau / ns ')

#%% show selected area in histogram
showSelectedArea(H, binNumber, intTauImage)


# %% get mask for M and L 
lipImage = getTauRangeImage(image,tau, tauRange=[200,400])
lipBinary = getMask(lipImage)

melImage = getTauRangeImage(image,tau, tauRange=[20,200])
melBinary = getMask(melImage)

viewer.add_labels(lipBinary, name='lip mask')
viewer.add_labels(melBinary*2, name= 'mel mask')


#%% detect objects
#M
melLabel, _nBlob1 = getGranule(melImage,melBinary,min_sigma=2, max_sigma=4, extraExpansion=2)
#L
lipLabel, _nBlob2 = getGranule(lipImage,lipBinary,min_sigma=3, max_sigma=5)

viewer.add_labels(melLabel)
viewer.add_labels(lipLabel)


#%% set ground true classes to objects
melGTImage, melGTClass = projectGroundTrueToLabels(classImage, melLabel)
lipGTImage, lipGTClass = projectGroundTrueToLabels(classImage, lipLabel)

GTImage = np.max(np.array([melGTImage, lipGTImage]), axis=0)

viewer.add_labels(GTImage, colormap=GrainId.colorNapari)


#%% get M granule Profiles

radius, intFit, tauFit = getProfiles(image, tau,melLabel, nPoly=2)

fig, ax = plt.subplots()

for ii,_color in enumerate(GrainId.colorMPL):
    ax.plot(radius,tauFit[melGTClass==ii+1,:].T,color=_color)
    ax.set_title('tau profile')
    ax.set_ylabel('tau /ns')
    ax.set_xlabel('radius ')


# %% plot parameters according the profile of M granule

maxInt = np.max(intFit,axis=1)
maxTau = np.max(tauFit,axis=1)
ratioInt = intFit[:,0]/maxInt
ratioTau = tauFit[:,-1]/tauFit[:,0]

'''
medTau = np.median(maxTau)
medInt = np.median(maxInt)
thrTau = 0.8
thrInt = 0.75
'''
#thrTauRatio = 1.15
thrTauRatio = 1.0


fig, ax = plt.subplots()

ax.scatter(ratioTau,ratioInt,s=20, color = GrainId.colorMPL[melGTClass-1])
ax.vlines(thrTauRatio, np.min(ratioInt),np.max(ratioInt),linestyles= ':')
ax.set_title('Classification criteria MelanoLipofuscin')
ax.set_ylabel('Int_centre / Int_max ')
ax.set_xlabel('Tau_edge / Tau_centre ')
ax.annotate('ML', xy= (thrTauRatio,1))

#%% according the criteria classify ML clusters from M
melFitClass = separateMLfromM(tauFit, thrTauRatio=thrTauRatio)

#%% show the classified granule

melFitImage = myClassToImage(melFitClass,melLabel)
lipFitImage = (lipLabel>0)*GrainId.name['L']

# add L M and ML together in one image
allFitImage = np.max(np.array([melFitImage, lipFitImage]), axis=0)

viewer.add_labels(allFitImage, colormap=GrainId.colorNapari)


# %%
napari.run()

plt.show()