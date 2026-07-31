# CIU Data Processor

The objective of this program is to process the Ion Mobility data into a format that is accepted by CIUSuite software. In certain conditions, such as presence of duplicated voltage values within the same .mzml file or .RAW folder, the data treatment within the CIUSuite software is impossible. This program aims to help with this issue. 

This program allows for the conversion of Waters Ion Mobility data from .RAW folder format to .mzml file via msconvert.exe.
Secondly, this program allows for the data extraction and arrangement for further treatment in CIUSuite software (to be imported as "Text Files (_raw.csv)").

MSConvert can be found here: https://proteowizard.sourceforge.io/
CIUSuite Software can be found here: https://github.com/RuotoloLab/CIUSuite2

Currently, the drift times obtained with this program does not match the values shown by MassLynx from Waters.

To install the program, it is sufficient to download the .py file.
