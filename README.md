# sap-flow-MCMC
This repository contains the codes to recreate the results from “Intra-specific variability in plant hydraulic parameters inferred from model inversion of sap flux data” published in JGR-Biogeosciences. The repository is organized as follows:


•	There are 9 python scripts corresponding to the main figures in the manuscript (Figure_XX.py).  Each figure uses data stored in a correspondingly named folder in this repository.  This stored data allows the user to not need to run the time consuming MCMC inversion.

•	The Data_preprocessing folder contains a script that formatted the meteorological and sap flux data taken from the University of Michigan Biological Station (US-UMBS) derived from the AmeriFlux (https://ameriflux.lbl.gov/sites/siteinfo/US-UMB ) and SAPFLUXNET (https://zenodo.org/record/3971689#.YlQpeOjMKUk) data products.

•	The MCMC folder contains a sample code that runs the Markov Chain Monte Carlo model inversion.  In the sample code, change the idx variable on Line 10 to select one of the 25 measurement sites (0-24). The results of the MCMC analysis are stored in the folder Figure_1&2&4  in pickle format.


These scripts were created with Python version 3.7.  Please contact the author with any questions
