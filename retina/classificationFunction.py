''' 
set of functions to perform classification of granules (M,L, ML) in flim image'''
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

#%% functions

class GrainId:
    name = {'BACKGROUND': 0,
            'L': 1,
            'M': 2,
            'ML':3,
            'NON':4}
    colorMPL = np.array(['red', 'green','blue','white'])
    colorNapari = {1:[1,0,0],2:[0,1,0],3:[0,0,1], 4:[1,1,1]}


def loadData(ffolder, imageFile,tauFile):
    # load measured data
    image = tifffile.imread(ffolder +'/' + imageFile)
    tau = pd.read_excel(ffolder +'/' + tauFile, sheet_name="TauMean 1", header=None).to_numpy()[::-1,:]
    
    #remove nan data
    tau[np.isnan(tau)] = 0

    return image, tau

def loadClassification(ffolder, classFile, imageSize, columnName= 'classification_Hala', sheet_name="Sheet1"):
    # load ground true
    _sheet = pd.read_excel(ffolder +'/' + classFile, sheet_name=sheet_name)
    gPos = np.vstack((_sheet['X'].to_numpy(),_sheet['Y'].to_numpy())).T.astype(int)

    _gClass = _sheet[columnName].to_list()
    gClass = np.array([GrainId.name[str.upper(ii.replace(" ", ""))] for ii in _gClass])

    classImage = np.zeros(imageSize,dtype=int)
    classImage[imageSize[0] - gPos[:,-1],gPos[:,-2]] = gClass
    #classImage[gPos[:,-2],gPos[:,-1]] = gClass


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

def getMaskOnTau(tau,tauRange, sigma= 3):
    ''' create mask from tau. apply smooth band pass filter
     return np.array of size tau, values from 0 to 1  '''
    fact1 = (erf((tau-tauRange[0])/sigma)+1)/2
    fact2 = (erf((tauRange[1]-tau)/sigma)+1)/2
    return fact1*fact2

def getTauRangeImage(image,tau,tauRange=[20,200], sigma=3):
    # get intensity image for certain tau range
    return image*getMaskOnTau(tau,tauRange)

def showSelectedArea(H, binNumber, intTauImage):
    ''' create two interactive napari viewer showing selected area of histogram in image'''

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


def getGranule(image,binary,max_sigma=5, min_sigma=2, 
               overlap=0.5, threshold=0.1,labelOffset=0, extraExpansion=0):
    ''' get labelled mask of the granule
    assumed granules are spherical mit min max radius'''
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
    labelImage = expand_labels(labels_data, distance=extraExpansion).astype(int)

    return labelImage, nBlob

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
            ax2.plot(radius,tauFit[ii,:], color = GrainId.colorMPL[myClass[ii]-1])

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

