GNSSjamLoc v0.1

# LICENSING
This file is part of GNSSjamLoc. 

GNSSjamLoc is a localization tool for intentional 
interfence sources (jammers), that are potentially disturbing
ligitimate signal reception (e.g. GPS). Given a dataset of power
field measurement associated with spatial coordinates and
distributed over an observation area, GNSSjamLoc can locate the
source of interference, even in complex propagation scenarios (e.g. urban).
It is based on a path loss physics-based model augmented with 
a data-driven component, i.e. a Neural Netrwork.
Additional information can be found at https://doi.org/10.48550/arXiv.2212.08097

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Copyright (C) 2023 Andrea Nardin <andrea.nardin@polito.it>
Navigation, Signal Analysis and Simulation (NavSAS) group,
Politecnico di Torino 

Copyright (C) 2023 Peng Wu, Tales Imbiriba, Pau Closas
Signal Processing Imaging Reasoning and Learning (SPIRAL) Lab
Northeastern University

If you use this software, please cite the relative paper as:
A. Nardin, T. Imbiriba, P. Closas, "Jamming Source Localization Using Augmented Physics-Based Model",
ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 
Rhodes, Greece, 2023, pp. -, doi: -.


# GNSSjamLoc

GNSSjamLoc is a localization tool for intentional 
interfence sources (jammers), that are potentially disturbing
ligitimate signal reception (e.g. GPS). Given a dataset of power
field measurement associated with spatial coordinates and
distributed over an observation area, GNSSjamLoc can locate the
source of interference, even in complex propagation scenarios (e.g. urban).
It is based on a path loss physics-based model augmented with 
a data-driven component, i.e. a Neural Netrwork.
Additional information can be found at https://doi.org/10.48550/arXiv.2212.08097

## jammedAgentsSimulation.m
This MATLAB program can simulate a dataset of power measurements distributed over an observation area.
Datapoints are generated with respect to jammer source characteristics, position, and dynamics.
Output data (X.mat,Y.mat, and trueJamLoc.mat) are exported and processed by the python code to estimate the jammer's location


## gnss-jam_loc.py
It estimates the jammer location from a dataset of power measurements and associated position information.


it runs over different INR, Monte Carlo, time instants
The hierarchy is :
	for each INR value
		for each MC realization
			for each time
				estimate the J. location


if time vector is > 1 but 'aggregated_over_time' is flagged, aggregated output data is computed both over MC simulations and time.

It is possible to obtain an aggregation effect over different random agents position by generating data for a static jammer over obs time > 1. 
Then by processing those data with aggregate_over_time=1 and optionally Nmc = 1;