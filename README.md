# Classi4RPE


Classi4RPE is a computational program to segment and classify the granules of Retinal Pigment Epithelium cells RPE
this classification is based on the Fluorescence lifetime measurements
 
Created in 2025
Wrtitten by: Ondrej Stranik, Maryam Ali, Rainer Heintzmann

It can read FLIM and intensity data for RPE measurements, and:
   - segment the granules after thresholding short/long lifetimes using seeded water shedding.
   - Identify Lipofuscins (Higher fluorescent)
   - Identify lower fluorescent granules and distiguish Malanolipouscins by computing their
   lifetime ratio from center to edge.
   - Export the segmented & classified granules data: coordinates, mean lifetimes.
   - Visualize selectied granules interactively by selecting the lifetime/intensity range from the histogram.

This code (including the setted parameters) has been tested on FLIM data sets from University Hospital Jena, Experimental Ophthalmology Group using Becker
& Hickl GmbH.
 
