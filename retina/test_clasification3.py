''' script to test clasification'''
#%%
import numpy as np
import napari
import matplotlib.pyplot as plt
import tifffile
import pandas as pd
#import glasbey
from skimage.color import hsv2rgb, rgb2hsv
from scipy.stats import binned_statistic_2d
from skimage.filters import threshold_otsu
from skimage.morphology import binary_erosion, binary_closing, binary_dilation
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.draw import disk
from skimage.segmentation import expand_labels
from scipy.special import erf
from scipy import ndimage as ndi

ffolder = r'G:\office\work\git\retina\DATA\2018004_1_1'
imageFile = '2018004_71_960nm_116fr_3Pro_1_1-Ch2-_intensity_image.tif'
tauFile = '2018004_1_1_tm_Ch2.xlsx'
#classFile = '2018004_1_1_ch2_classification.ods'
#columnName = 'classification'
classFile = '2018004_1_1_ch2_classification2.xlsx'
columnName = 'classification_Hala'

labelFile = '2018004_1_1_ch2_labels.txt'

class GrainId:
    name = {'BACKGROUND': 0,
            'L': 1,
            'M': 2,
            'ML':3,
            'NON':4}

classColorNP = np.array(['white', 'green','blue','red','white'])
myColorMap = {1:[0,1,0],2:[0,0,1],3:[1,0,0], 4:[1,1,1]}



#%% functions

def loadData(ffolder, imageFile,tauFile):
    # load measured data
    image = tifffile.imread(ffolder +'/' + imageFile)
    tau = pd.read_excel(ffolder +'/' + tauFile, sheet_name="TauMean 1", header=None).to_numpy()[::-1,:]
    
    #remove nan data
    tau[np.isnan(tau)] = 0

    return image, tau

def loadClassification(ffolder, classFile, columnName= 'classification_Hala', sheet_name="Sheet1" ):
    # load ground true
    _sheet = pd.read_excel(ffolder +'/' + classFile, sheet_name=sheet_name)
    gPos = np.vstack((_sheet['X'].to_numpy(),_sheet['Y'].to_numpy())).T.astype(int)

    _gClass = _sheet[columnName].to_list()
    gClass = np.array([GrainId.name[str.upper(ii)] for ii in _gClass])

    classImage = np.zeros_like(image)
    classImage[gPos[:,-1],gPos[:,-2]] = gClass

    return classImage

def loadMLabel(ffolder, labelFile):
    # load Maryam segmentation
    givenLabel = pd.read_csv(ffolder +'/' + labelFile, sep='\t').to_numpy()
    return givenLabel


def getTauIntensityImage(image,tau,tauRange= [50,1000]):
    # get tauIntensity Image
    tauScaled = (tau-tauRange[0])/tauRange[1]
    tauScaled[tauScaled<0] = 0
    tauScaled[tauScaled>1000] = 1

    _intTauImage = np.array((tauScaled,np.ones_like(image),image/np.max(image)))
    intTauImage = np.swapaxes(np.swapaxes(hsv2rgb(_intTauImage, channel_axis = 0),0,2),0,1)
    return intTauImage


def getTauIntensityHistogram(image,tau, tauRange=[50,1000], bins=200):
    ''' get 2D histogram'''
    _tau = np.copy(tau)
    # remove nan value
    _tau[np.isnan(_tau)] = 0

    sel = (tau<tauRange[0]) | (tau>tauRange[1])
    _tau[sel] = 0
    _tau = _tau.flatten()
    _image = np.copy(image)
    _image = _image.flatten()

    H, xEdge, yEdge, binnumber = binned_statistic_2d(
    _tau,_image, None, bins=bins, range = [tauRange,[0, np.max(_image)]],statistic='count',expand_binnumbers=True)
    H = H.T
    return H, xEdge, yEdge, binnumber

def getMask(image, extra_factor=1, minSize=3):
    #get mask
    thresh = threshold_otsu(image)
    binary = image > thresh*extra_factor

    binary = binary_erosion(binary, footprint= np.ones((minSize, minSize)))
    binary = binary_dilation(binary, footprint= np.ones((minSize, minSize)))

    return binary

def bandPassFilter(tau,tauRange, sigma= 3):
    fact1 = (erf((tau-tauRange[0])/sigma)+1)/2
    fact2 = (erf((tauRange[1]-tau)/sigma)+1)/2
    return fact1*fact2

def getTauRangeImage(image,tau,tauRange=[20,200], sigma=3):
    # get intensity image for certain tau range
    return image*bandPassFilter(tau,tauRange)

def showSelectedArea(H, binNumber, intTauImage):

    def updateImage():
        print('image updated')
        try:
            _label = viewer2.layers['area'].to_labels(H.shape)
            viewer2.layers['labels'].data = _label
            shapeIdx = binNumber.reshape((2,*intTauImage.shape[:2]))
            idxInImage = _label[shapeIdx[1,...]-1,shapeIdx[0,...]-1]
            viewer.layers['idxInImage'].data = idxInImage
        except:
            print('not updated')

    viewer = napari.Viewer()
    viewer.add_image(intTauImage, rgb=True)
    viewer.add_labels(np.zeros(intTauImage.shape[:2], dtype=int), name='idxInImage')

    viewer2 = napari.Viewer()
    viewer2.add_image(H, name='2D-histogram', colormap='hsv')
    viewer2.add_shapes(name= 'area')
    viewer2.add_labels(np.zeros_like(H, dtype=int),name= 'labels')
    viewer2.layers['area'].events.data.connect(updateImage)


def getGranule(image,binary,max_sigma=5, min_sigma=2, overlap=0.5, threshold=0.1,labelOffset=0):
    # get position of granule and its size
    blobs_log = blob_log(image*binary, 
                         max_sigma=max_sigma, 
                         min_sigma=min_sigma, 
                         overlap=overlap,
                         threshold=threshold,
                         exclude_border=True)
    labels_data = np.zeros_like(image)
    nBlob = len(blobs_log[:])
    for ii,_blob in enumerate(blobs_log):
        rr, cc = disk((_blob[0], _blob[1]), _blob[2], shape=image.shape)
        labels_data[rr, cc] = ii+1 + labelOffset  

    # extra expand the label by one
    expanded = expand_labels(labels_data, distance=1).astype(int)

    return expanded, nBlob

def projectGroundTrueToLabels(classImage,labelImage):
    ''' project ground true classes to the labels'''
    nBlob = np.max(labelImage)
    _myClass = np.zeros(nBlob+1,dtype=int) +4
    _myClass[labelImage[classImage>0]] = classImage[classImage>0].astype(int)
    #set zero index to background 
    _myClass[0] = 0
    # get the image of blobs with ground true classification
    myClassImage = np.zeros_like(labelImage)
    myClassImage = _myClass[labelImage]

    # remove background index
    myClass = _myClass[1:]

    return myClassImage, myClass

def getProfiles(image,tau,expanded, nPoly=2, showData= False, myClass=None):
    ''' nPoly ... order of the polynomial fit'''
    distance = ndi.distance_transform_edt(expanded>0)
    #viewer.add_image(distance, name='distanced')

    # nR ... resolution in radial distance of the spot
    nR = 50

    nBlob = np.max(expanded)

    radius = np.linspace(0,1,nR)
    tauFit = np.zeros((nBlob,nR))
    intFit = np.zeros((nBlob,nR))

    if showData:
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()

    for ii in range(nBlob):
        mask = expanded==ii+1
        intIdx = image[mask]
        tauIdx = tau[mask]

        _disIdx = distance[mask]
        disMax = np.max(_disIdx)
        disIdx = (disMax - _disIdx)/(disMax-1)
        
        # intensity fit
        z = np.polyfit(disIdx, intIdx, nPoly)
        intFit[ii,:] = np.poly1d(z)(radius)

        # tau fit
        z = np.polyfit(disIdx, tauIdx, nPoly)
        tauFit[ii,:] = np.poly1d(z)(radius)


        if showData:
            if myClass is None:
                myClass = np.ones(nBlob)
            ax1.scatter(disIdx.ravel(),intIdx.ravel(),color = classColorNP[myClass[ii]])
            ax1.plot(radius,intFit[ii,:],color = classColorNP[myClass[ii]])
            ax2.scatter(disIdx.ravel(),tauIdx.ravel(),color = classColorNP[myClass[ii]])
            ax2.plot(radius,tauFit[ii,:], color = classColorNP[myClass[ii]])

    return radius, intFit, tauFit

def separateMLfromM(tauFit, thrTauRatio = 1):
    ''' separate ML from M according the tau profile'''
    maxTau = np.max(tauFit,axis=1)
    ratioTau = tauFit[:,-1]/tauFit[:,0]

    # set everything to M
    myFitClass = np.zeros_like(maxTau,dtype=int)
    myFitClass[:] = GrainId.name['M']

    myFitClass[ratioTau > thrTauRatio] = GrainId.name['ML']

    return myFitClass

def myClassToImage(myFitClass, labelImage):
    ''' make image of granule colored with a classification'''
    _myFitClass= np.hstack(([0],myFitClass))
    myClassFitImage = np.zeros_like(labelImage)
    myClassFitImage = _myFitClass[labelImage]

    return myClassFitImage


#%% process data

image, tau = loadData(ffolder, imageFile,tauFile)
classImage = loadClassification(ffolder,classFile)
intTauImage = getTauIntensityImage(image,tau)

# show original data
# original glasbey
#glas = glasbey.create_palette(256)
viewer = napari.Viewer()
viewer.add_image(image, name='intensity')
viewer.add_image(intTauImage, rgb=True)
viewer.add_labels(classImage,colormap=myColorMap)

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
melLabel, _nBlob1 = getGranule(melImage,melBinary,min_sigma=2, max_sigma=4)
#L
lipLabel, _nBlob2 = getGranule(lipImage,lipBinary,min_sigma=2, max_sigma=4)

viewer.add_labels(melLabel)
viewer.add_labels(lipLabel)


#%% set ground true classes to objects
melGTImage, melGTClass = projectGroundTrueToLabels(classImage, melLabel)
lipGTImage, lipGTClass = projectGroundTrueToLabels(classImage, lipLabel)

GTImage = np.max(np.array([melGTImage, lipGTImage]), axis=0)

viewer.add_labels(GTImage, colormap=myColorMap)


#%% get M granule Profiles

radius, intFit, tauFit = getProfiles(image, tau,melLabel, nPoly=2)

# plot the profiles
fig1, ax1 = plt.subplots()
for ii,_color in enumerate(classColorNP):
    ax1.plot(radius,tauFit[melGTClass==ii,:].T,color=_color)
ax1.set_title('tau profile')
ax1.set_ylabel('tau /ns')
ax1.set_xlabel('radius ')


# %% plot parameters according the profile of M granule

maxInt = np.max(intFit,axis=1)
maxTau = np.max(tauFit,axis=1)
ratioInt = intFit[:,0]/maxInt
ratioTau = tauFit[:,-1]/tauFit[:,0]

medTau = np.median(maxTau)
medInt = np.median(maxInt)
thrTau = 0.8
thrInt = 0.75
thrTauRatio = 1.15

fig3, ax3 = plt.subplots()
ax3.scatter(ratioTau,ratioInt,s=20, color = classColorNP[melGTClass])
ax3.vlines(thrTauRatio, np.min(ratioInt),np.max(ratioInt),linestyles= ':')
ax3.set_title('Classification criteria MelanoLipofuscin')
ax3.set_ylabel('Int_centre / Int_max ')
ax3.set_xlabel('Tau_edge / Tau_centre ')
ax3.annotate('ML', xy= (thrTauRatio,1))

#%% according the criteria classify ML clusters from M
melFitClass = separateMLfromM(tauFit, thrTauRatio=thrTauRatio)

#%% show the classified granule

melFitImage = myClassToImage(melFitClass,melLabel)
lipFitImage = (lipLabel>0)*GrainId.name['L']

allFitImage = np.max(np.array([melFitImage, lipFitImage]), axis=0)

viewer.add_labels(allFitImage, colormap=myColorMap)

# set everything to M
#viewer.add_labels((expanded2>0), colormap=myColorMap)


# %%
napari.run()

plt.show()