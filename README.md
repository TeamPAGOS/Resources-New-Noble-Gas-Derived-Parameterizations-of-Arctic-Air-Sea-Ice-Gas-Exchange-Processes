# Resources-New-Noble-Gas-Derived-Parameterizations-of-Arctic-Air-Sea-Ice-Gas-Exchange-Processes
Resources for data and plot generation for paper New Noble Gas-Derived Parameterizations of Arctic Air-Sea-Ice Gas Exchange Processes (Hubner et al., in preparation).

The file `Paper copy.ipynb` and `transient_tracer.py` should work more or less out-of-the-box. One step is necessary, which is extracting `data/SO21_ctd_noheader.zip` into the `data/` directory, so that the tab file is accessible. It is zipped here due to Github's upload size constraints.

In `montecarlo.py`, if you really want to run it, then you must set the output directories for the monte carlo fits, as indicated by the placeholder pathnames in the file.
