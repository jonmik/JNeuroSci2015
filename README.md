NatNeuro2015
============
% JMM_NN2015_POC A POC acompannying Leonard, et al. (2015) in NatNeuro.
    %
    % The following file is a proof of concept demonstration 
    % meant to acompany: 
    % 
    % Leonard, et al. (2015). Sharp wave ripples during visual exploration in the primate hippocampus
    %
    % Author: Jonathan Mikkila, Jan-2015
    %
    % About:
    %  The goal of the analysis is to investigate if there are differences in
    %  time-frequency characteristics between different classes in a data driven way to
    %  corroborate the assignment of ontological categories (or 'types') which researchers
    %  prescribe to neural events. It asks the question - 'is there
    %  something that it is to be a [-label-]?' 
    %
    %  The proof of concept is provided in such a way that it is meant to be
    %  immediately applicable as a tool to the users own experimental data. a minimal
    %  collection of graphical output is provided to orientate the user, but we
    %  encourage users to extend this code to suit their own purposes.
    %
    % Dependencies:
    %  Apart from the dependencies provided in the folder /include/, this class
    %  also assumes the user has the Neural Network Toolbox and Statistics
    %  Toolbox from MathWorks available to them. If you do not, there exist more
    %  powerful, involved, and open source implementations of self organizing (Kohonen) maps,
    %  and non-negative matrix factorization through toolboxes which can be readily
    %  substituted into this file - see the following references:
    %
    %     Vesanto, et al. (2000). SOM toolbox for Matlab 5.
    %     Helsinki: Helsinki University of Technology.
    %
    %     Li, & Ngom. (2013). The non-negative matrix factorization toolbox for biological data mining.
    %     Source Code for Biology and Medicine, 8, 10. doi:10.1186/1751-0473-8-10
    %
    % Usage:
    %  Make sure to add the include directory to your search path.
    %  Configuration Variable assignment have been commented out, please
    %  uncomment and read descriptions before continuing (lines 125-160). 
    %
    % Input:
    %
    %  The usage of this class is straight forward. It assumes data is provided
    %  from 2 or more user-defined, equally sized categories of events. The data here
    %  are assumed to be spectrograms, but this does not strictly have to be the case.
    %  What is important is that at whichever scale the user has chosen,
    %  differences between individual events are pronounced and distinguishable, 
    %  as well as centred in a meaningful and accurate way.
    %
    %  Input is to be formatted as an array of structures
    %  with the following fields:
    %    entry(i).spectrogram =  [NxM double];
    %    entry(i).labelID     =  double;
    %    entry(i).label       = 'string';
    %
    %    '.labelID' should be an enumerated identifier, while '.label' is intended to
    %    be a human readable string.
    %
    % Configuration:
    %  The settings(/configuration variables) for the analysis are entirely defined
    %  by the public properties. These can be set either by modifying the code,
    %  creating an instance of the class and pragmatically changing the properties
    %  as you would with a struct, or providing a structure with identical field names
    %  with the values you wish to use.
    %
    %  The number of labels is determined from the input and has no
    %  configuration variable.
    %
    % Output:
    %  All output is collected under the local folder
    %   cd()\JMM_NN2015_POC_output\<dateVec>\
    %  The program will output classes as they are defined by the self organizing
    %  map. The number of classes, and how closely they fit the data as a whole
    %  are properties of the self organized map, which we leave to the user to
    %  configure through it's parameters. It will also output a series of images, 
    %  the first of which is a scatter plot with the maximum class specificity along any 
    %  one label, as well as the confidence intervals for each p-val specified. 
    %  In addition to this image, the class which is most specific to each enumerated label
    %  will also have the spectrogram for its best matching unit, and a centroid
    %  based on the feature weights of its members (via time-frequency non-negative
    %  matrix factorization). In the title of this figure will appear the
    %  label most represented; the class id; the composition by percent, and size. 
    %  Lastly a file detailing each class will be saved with the name matFile.mat
    %  The variable therein will be an object from this class.

    %  Properties with public get access can be used to retrieve results
    %  from properties with private set access (near end of file) 
    %  as if they were a struct - eg:
    % 
    %     pocObj = JMM_NN2015_POC(input, settings)
    %     classificationResults = pocObj.classStructureArray;
    %
    % Output Fields for classStructureArray:
    %  .indices                        - indices to class members from original input 
    %  .significance                   - p-val < based on distance from label swap.
    %  .weightCentroid                 - weight for product with the object field .spectralFeatures
    %  .centroidSpectrogramFromWeights - spectrogram construction from weights (wrt)
    %  .bestMatchedMemberIndex         - index for input closest to SOM unit
    %  .bestMatchedSpectrogram         - copy of structureIn(bestMatchedMemberIndex).spectrogram
    %  .classComposition               - representation from each labelID in corresponding input
    %  .size                           - number of elements represented by class
    %  .classDistance                  - euclidean distance from the midpoint in composition space
    %  .classID                        - enumerated identifier 
    %
    % Program Layout:
    %  The POC is largely self-contained to the class. If you are not familiar
    %  with classes, begin at the function JMM_NN2015_POC(...).
    %
    %  The program flow has been restricted to Main(...), variables and
    %  functions have been given descriptive and verbose titles.
    %
    %  See the public properties near the top of the class for settings which
    %  must be configured, along with their descriptions and some recommended ranges.
    %
    %
    %  Please feel free to contact me at JonMik@YorkU.Ca with any questions.
    %
    %  This proof of concept is supplied without warranty.
    %  Please consider citing this paper if you extend this utility.
