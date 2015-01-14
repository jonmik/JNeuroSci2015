classdef JMM_NN2015_POC < handle
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
    
    
    properties (Access = public)
        
        %% Configuration Variables
        
        autoStart     = %true; % set to true to begin processing on object construction
        % false to allow for programatic parameterization via public properties before calling Start();
        
        graphicOutput = %true; % set to true to generate example figures of most label-specific class
        % in a local date labeled directory. Code provided as proof of concept,
        % intended to allow users to extend for their own needs.
        
        fileOutSave   = %true; % set to true to generate a .mat file containing the object and classStructure output.
        
        
        %% NNMF configuration
        numberOfSpectralFeatures = %[5]; %leave empty to estimate empirically (time consuming)
        numberOfEstimateAttempts = %1;   %Number of attempts to minimize error term in low-rank aproximation ('feature extraction'), wrt above.
        maxSpectralFeatureCount  = %10;  %Maximum number of features to estimate, wrt above.
        
        %% Machine Learning Configuration
        % Strong user discretion advised: parameters should fit data and be
        % based on sensible observations and followed up with examanation
        % of results.
        
        mapDimensions   = %[4 4]; % Non-zero vector of non-zero dimensions, will determine how many classes will be produced.
        mapNeighborhood = %1;     % Neighborhood Size: suggested range [0, 2], how many degrees of seperation are effected by each weight updated.
        
 	% See Flexer. (1997). Limitations of self-organizing maps for vector quantization and multidimensional scaling.
        % If user is uncomfortable with fixed neighborhood training, see http://www.cis.hut.fi/projects/somtoolbox/about
        
        
        %% Shuffle Permutation
        
        numberofShuffles  = %2000; % number of permutation shuffles
        ciArray           = %[0.05 0.01 0.001]; %p values to use
        
        
    end
    
    methods (Access = public)
        
        function this = JMM_NN2015_POC(structureIn,varargin)
            
            this.structureIn = structureIn;
            
            if nargin > 1
                
                this.LoadSettings(varargin{1});
                
            end
            
            if this.autoStart
                
                this.Start();
                
            end
        end
        
        function this = Start(this)
            
            this.outDir   = [pwd '\JMM_NN2015_POC_output\' datestr(now,30) '\'];
            this.imageDir = [this.outDir 'images\'];
            this.fileDir  = [this.outDir 'matFiles\'];
            
            this.Main();
            
        end
        
        function this = LoadSettings(this,settingStruct)
            
            localNames = fieldnames(this);
            inNames    = fieldnames(settingStruct);
            
            inNamesIdx = inNames(ismember(inNames,localNames));
            
            for i = find(~inNamesIdx)
                warning(['JMM_NN2015_POC::LoadSettings(...), ' inNames{i} ' not valid property, skipping assignment.']);
            end
            
            for i = find(inNamesIdx)
                this.(inNames{i}) = settingStruct(inNames{i});
            end
            
        end
        
    end
    
    methods (Access = private)
        
        function this = Main(this)
            
            structureIn = this.structureIn; %Attach from function to object
            
            spectrogramRowMatrix = this.FlattenSpectrogramStructure(structureIn); %convert spectrograms into vectors
            
            % determine number of features if not provided by user
            if ~isempty(this.numberOfSpectralFeatures) 
                featureEstimateCount = this.numberOfSpectralFeatures;
            else
                featureEstimateCount = this.EstimateFeatureCount(spectrogramRowMatrix); 
            end
            
            % estimate time-frequency features using NNMF
            featuresStruct = this.EstimateSpectralFeatures(spectrogramRowMatrix, featureEstimateCount);
            
            % apply unsupverised learning to time-frequency features
            classMembershipAndFeatureStruct = this.LearnClassMembership(featuresStruct);
            
            % assign results to structure array
            structureIn = this.AssignClassMembership(structureIn, classMembershipAndFeatureStruct);
            
            % produce a set of indicies to use for label swappig
            shuffleIndicieMatrix = this.PermuteIndicies(numel(structureIn));
            
            % determine ranked order distances and class specificity 
            this.ComputeConfidenceIntervalsOnLabelShuffledData(structureIn, shuffleIndicieMatrix);
            
            % produces and populates structures with fields describing each class
            classStructureArray = this.buildClassStructureWithStatistics(structureIn);
            
            % save results to object properties
            this.classStructureArray = classStructureArray;
            this.structureIn         = structureIn;
            
            this.complete = true;
            
            % save object to matfile
            this.SaveMatFiles();
            
            % produce example figures, then save them
            this.OutputFigures();
            
        end
        
        function spectrogramRowMatrix = FlattenSpectrogramStructure(this, spectrogramStructure)
            
            structSize = numel(spectrogramStructure);
            
            [freqBinCount, timeSampleLength] = size(spectrogramStructure(1).spectrogram);
            
            this.spectrogramDimM = freqBinCount;
            this.spectrogramDimN = timeSampleLength;
            
            flattenedLength = freqBinCount .* timeSampleLength;
            
            spectrogramRowMatrix = zeros(structSize, flattenedLength);
            
            for i = 1:structSize
                spectrogramRowMatrix(i,:) = reshape(spectrogramStructure(i).spectrogram, 1, flattenedLength);
            end
            
            
        end
        
        function reShapedOutput = UnFlattenSpectrogramRowMatrix(this, spectrogramArrayInput)
            
            if isa(spectrogramArrayInput, 'double')
                
                if all(size(spectrogramArrayInput) > 1)
                    error('JMM_NN2015_POC::UnFlattenSpectrogramRowMatrix(...) - invalid input, see method definition');
                end
                
                
                reShapedOutput = reshape(spectrogramArrayInput, [this.spectrogramDimM this.spectrogramDimN]);
                
            elseif isa(spectrogramArrayInput, 'double')
                
                structSize = numel(spectrogramArrayInput);
                
                for i = 1:numel(spectrogramArrayInput)
                    spectrogramArrayInput.spectrogram = reshape(spectrogramArrayInput, [this.spectrogramDimM this.spectrogramDimN]);
                end
                
                reShapedOutput = spectrogramArrayInput;
                
            end
            
        end
        
        function estimateNNMF_rank = EstimateFeatureCount(this, spectrogramRowMatrix)
            
            MIN_FEATURES        = 2;    % Arbitrary small integer greater than one, recommended value.
            MAX_NNMF_ITERATIONS = 1000; % Arbiraty large value, will converge long before in most cases.
            
            seed_ = RandStream('mt19937ar', 'Seed', 0);
            RandStream.setGlobalStream(seed_);
            options = statset('Streams', seed_, 'MaxIter', MAX_NNMF_ITERATIONS);
            
            estimateErrorArray = zeros(1,1 + this.maxSpectralFeatureCount - MIN_FEATURES);
            
            for i = MIN_FEATURES:this.maxSpectralFeatureCount
                [~, ~, estimateErrorArray(1 + i - MIN_FEATURES)] =    ...
                    nnmf(spectrogramRowMatrix, i, 'algorithm', 'als', ...
                    'replicates', i, 'options', options         );
            end
            
            [~,minRankIdx] = min(estimateErrorArray);
            estimateNNMF_rank = -1 + minRankIdx + MIN_FEATURES;
            
            if i == this.maxSpectralFeatureCount
                disp('JMM_NN2015_POC::EstimateFeatureCount() Reached max feature count, consider increasing parameter');
            end
            disp(['Feature count: ' int2str(estimateNNMF_rank) '. Save and enter as a parameter for this dataset  '...
                'to speed up subsequent analysis']);
            
        end
        
        
        function membershipAndFeatureStruct = LearnClassMembership(this, featuresStruct)
            
            MAX_SOM_PASSES = 500; % Arbiraty large value, will converge long before in most cases.
            
            featureDimN = [featuresStruct.W];
            featureDimM = [featuresStruct.H];
            
            dimension2Learn = [];
            
            if any(size(featureDimN) == numel(this.structureIn) )
                
                dimension2Learn       = featureDimN;
                this.spectralFeatures = featureDimM;
                
            else
                
                dimension2Learn       = featureDimM;
                this.spectralFeatures = featureDimN;
                
            end
            
            if size(dimension2Learn,2) < size(dimension2Learn, 1)
                dimension2Learn = dimension2Learn';
            end
            
            selfOrgMapNetwork = selforgmap(this.mapDimensions, MAX_SOM_PASSES, 0, 'gridtop', 'dist');
            networkClassifier = train(selfOrgMapNetwork, dimension2Learn);
            
            indicies = vec2ind(networkClassifier(dimension2Learn) );
            
            membershipAndFeatureStruct = cell(1, numel(indicies) );
            
            
            for i = 1:numel(membershipAndFeatureStruct)
                
                newStructEntry = [];
                
                newStructEntry.featureWeights = featureDimN(i, :);
                newStructEntry.membership     = indicies(i);
                
                membershipAndFeatureStruct{i} = newStructEntry;
                
            end
            
            membershipAndFeatureStruct = cell2mat(membershipAndFeatureStruct);
            
        end
        
        function shuffleIndicieMatrix = PermuteIndicies(this,numEntries)
            
            seed_ = RandStream('mt19937ar', 'Seed', 2);
            RandStream.setGlobalStream(seed_);
            
            % Uniform random resampling indicies with replacement.
            shuffleIndicieMatrix = floor(rand([this.numberofShuffles,numEntries]) .* (numEntries )) + 1;
            
        end
        
        function this = ComputeConfidenceIntervalsOnLabelShuffledData(this, inStructure, shuffleIndicieMatrix)
            
            EuclideanDistance = @(x, y) sum((x - y) .^2) .^ 0.5;
            
            [labelIDs, ~ , labelCounts] = unique([inStructure.labelID]);
            
            if ~all(logical(diff(labelCounts)))
                warning(['JMM_NN2015_POC - user input violates assumption of equal number   ' ...
                         'of elecments for each set of labels, large deviations will        ' ...
                         'produce invalid results.']                                        );
            end
            
            midPoint = repmat(1./numel(labelIDs), 1, numel(labelIDs) );
            
            shuffledStructres = inStructure(shuffleIndicieMatrix);
            
            shuffledCompositions = zeros( size( shuffleIndicieMatrix, 1), numel(labelIDs) );
            
            shuffledSpecificity  = zeros(size( shuffleIndicieMatrix, 1), 1);
            
            shuffledDistancesFromMidPoint = zeros(size( shuffleIndicieMatrix, 1), 1);
            
            for i = 1:size(shuffleIndicieMatrix, 1)
                for j = 1:numel(labelIDs)
                    shuffledCompositions(i, j) = sum( [shuffledStructres(i,:).labelID] == labelIDs(j) );
                end
                
                shuffledCompositions(i, :) = shuffledCompositions(i, :) ./ sum(shuffledCompositions(i, :));
                shuffledSpecificity(i)    = max(shuffledCompositions(i, :));
                
                shuffledDistancesFromMidPoint(i, 1) = EuclideanDistance(shuffledCompositions(i,:), midPoint);
            end
            
            rankedDistances   = sort(shuffledDistancesFromMidPoint, 'descend');
            rankedSpecificity = sort(shuffledSpecificity, 'descend');
            
            for i = 1:numel(this.ciArray)
                this.ciValues(i) = rankedDistances(ceil(numel(rankedDistances) * this.ciArray(i)));
                this.csValues(i) = rankedSpecificity(ceil(numel(rankedSpecificity) * this.ciArray(i)));
            end
            
        end
        
        function classStructureArray = buildClassStructureWithStatistics(this,structureIn)
            
            EuclideanDistance = @(x, y) sum((x - y) .^2) .^ 0.5;
            
            classIDs       = unique([structureIn.classID]);
            
            [this.labelIDs, labelIdx] = unique([structureIn.labelID]);
            this.labelText = {structureIn(labelIdx).label};
            
            midPoint = repmat(1./numel(this.labelIDs), 1, numel(this.labelIDs));
            
            classStructureArray = cellmat(1, numel(classIDs));
            
            for i = 1:numel(classIDs)
                
                classStruct = [];
                
                classStruct.classID = i;
                
                classStruct.indices = find([structureIn.classID] == classIDs(i));
                
                classInstances = structureIn(classStruct.indices);
                
                classComposition = zeros(1, numel(this.labelIDs));
                
                for j = 1:numel(this.labelIDs)
                    classComposition(1, j) = sum([classInstances.labelID] == this.labelIDs(j));
                end
                
                classComposition = classComposition ./ sum(classComposition);
                
                classDistance = EuclideanDistance(classComposition, midPoint);
                
                classStruct.significance = NaN;
                
                for k = 1:numel(this.ciValues)
                    if classDistance > this.ciValues(k);
                        classStruct.significance = this.ciArray(k);
                    end
                end
                
                featureWeights = vertcat(classInstances.featureWeights);
                
                classStruct.weightCentroid    = mean(featureWeights, 1);
                classStruct.centroidSpectrogramFromWeights =  this.UnFlattenSpectrogramRowMatrix(                   ...
                                                                 classStruct.weightCentroid * this.spectralFeatures );
                
                featureWeightMahalDistancesFromCentroid = mahal(featureWeights, featureWeights);
                
                [~, classStruct.bestMatchedMemberIndex] = min(featureWeightMahalDistancesFromCentroid);
                
                
                classStruct.bestMatchedSpectrogram = structureIn(classStruct.bestMatchedMemberIndex).spectrogram;
                
                classStruct.classComposition = classComposition;
                classStruct.size             = numel(classInstances);
                classStruct.classDistance    = classDistance;
                
                classStructureArray{i} = classStruct;
                
            end
            
            classStructureArray = cell2mat(classStructureArray);
            
        end
        
        function this = SaveMatFiles(this)
            
            if ~this.fileOutSave
                return;
            end
            
            if ~exist(this.outDir,'dir')
                mkdir(this.outDir);
            end
            
            if ~exist(this.fileDir,'dir');
                mkdir(this.fileDir);
            end
            
            JMM_NN2015_Obj = this;
            save([ this.fileDir 'matFile.mat'], 'JMM_NN2015_Obj', '-v6');
            
        end
        
        function this = OutputFigures(this)
            
            if ~this.graphicOutput
                return;
            end
            
            if ~exist(this.outDir,'dir')
                mkdir(this.outDir);
            end
            
            if ~exist(this.imageDir,'dir');
                mkdir(this.imageDir);
            end
            
            this.GenerateSpecificityFigure();
            this.Generate_N_SpecificityExamples();
            
        end
        
        function this = GenerateSpecificityFigure(this)
            
            NUMBER_OF_COLORS = 66;
            close all;
            
            figure('units', 'normalized', 'outerposition', [0 0 1 1]);
            
            hold all;
            
            classComposition = vertcat(this.classStructureArray.classComposition);
            
            [maxCompositionComponent, ...
                maxCompositionComponentIndex] = sort( max(classComposition, [], 2) ,'descend' );
            
            classSizes = [this.classStructureArray.size];
            classSizes = classSizes(maxCompositionComponentIndex);
            
            colorMap_      = repmat(zeros(1, 3), NUMBER_OF_COLORS, 1);
            colorMap_(:,1) = linspace(0, 1, NUMBER_OF_COLORS);
            
            colorSet  = floor(maxCompositionComponent .* NUMBER_OF_COLORS);
            colorSet  = colorMap_(colorSet,:);
            
            hold all
            
            scatter(1:numel(maxCompositionComponent), maxCompositionComponent, classSizes, colorSet, 'fill');
            
            ylim([.33, 1]);
            xlim([1, numel(maxCompositionComponent)]);
            
            set(gca, 'xtick', 1:numel(maxCompositionComponent));
            set(gca, 'xticklabel', maxCompositionComponentIndex);
            
            set(gca, 'ytick', [.33 this.csValues 1]);
            
            significanceLabels = cell(1, numel(this.ciValues));
            
            for j = 1:numel(this.ciValues)
                significanceLabels{1, j} = [' P < ' num2str(this.ciArray(j), 3)];
                plot(1:numel(maxCompositionComponent), repmat(this.csValues(j), 1, numel(maxCompositionComponent)), '-.k');
            end
            
            set(gca, 'ytickLabel', {'.33',significanceLabels{:}, '1'})
            
            xlabel('Class ID');
            ylabel('Class Max Specificity');
            
            export_fig([this.imageDir 'SpecificitySummaryFigure.jpg'], '-a1', '-q100');
        end
        
        function this = Generate_N_SpecificityExamples(this)
            
            if ~this.complete
                this.Start();
            end
            
            classCompostionArray = vertcat(this.classStructureArray.classComposition);
            [~, maxSpecificityIdx] = max(classCompostionArray, [], 1);
            
            for i = 1:numel(this.labelIDs)
                
                close all;
                
                classCompositionAproximation = this.classStructureArray(maxSpecificityIdx(i)).classComposition .* 100;
                
                classCompositionAproximationString = cell(1,numel(this.labelIDs));
                
                for j = 1:numel(classCompositionAproximation)
                    classCompositionAproximationString{1,j} = [this.labelText{j} ' ' num2str(classCompositionAproximation(j),2) '%; '];
                end
                
                classCompositionAproximationString = cell2mat(classCompositionAproximationString);
                
                classInstance = this.classStructureArray(maxSpecificityIdx(i));
                
                figure('units', 'normalized', 'outerposition', [0 0 1 1]);
                
                
                subplot(1,2,1)
                
                sanePColor(classInstance.centroidSpectrogramFromWeights                       ./  ...
                    sum( sum( classInstance.centroidSpectrogramFromWeights))                      );
                
                title(['Spectrogram from class defined by machine learning ' char(10) 'and feature composition (not from data).'])
                
                set(gca,'xtick',[],'ytick',[]);
                
                
                subplot(1,2,2)
                
                sanePColor(classInstance.bestMatchedSpectrogram                               ./ ...
                    sum( sum( classInstance.centroidSpectrogramFromWeights))                      );
                
                title(['Spectrogram for most representative element from data.'])
                
                suptitle(['Label Example ' this.labelText{i} ' (' int2str(this.labelIDs(i)) ') '    ...
                    ' classID: ' int2str(maxSpecificityIdx(i))                                      ...
                    char(10) 'composition by label - ' classCompositionAproximationString           ...
                    char(10) 'size - ' int2str(classInstance.size) ]                                );
                
                set(gca,'xtick',[],'ytick',[]);
                
                export_fig([this.imageDir 'LabelSpecificClassExample_L' int2str(i) '.jpg'], '-a1', '-q100');
                
            end
            
        end
        
    end
    
    methods (Static, Hidden = true, Access = private)
        
        function featuresStruct = EstimateSpectralFeatures(spectrogramRowMatrix, featureEstimateCount)
            
            MAX_NNMF_ITERATIONS = 1000; % Arbiraty large value, will converge long before in most cases.
            
            seed_ = RandStream('mt19937ar', 'Seed', 1);
            RandStream.setGlobalStream(seed_);
            options = statset('Streams', seed_, 'MaxIter', MAX_NNMF_ITERATIONS);
            
            [featuresStruct.W, featuresStruct.H] = ...
                nnmf(spectrogramRowMatrix, featureEstimateCount, 'algorithm', 'als', 'options', options);
            
        end
        
        
        function updatedStructure = AssignClassMembership(inStructure, classMembershipArray)
            
            updatedStructure = cell(1,numel(inStructure));
            
            for ij = 1:numel(inStructure)
                newEntry = inStructure(ij);
                
                newEntry.classID        = classMembershipArray(ij).membership;
                newEntry.featureWeights = classMembershipArray(ij).featureWeights;
                
                updatedStructure{1,ij} = newEntry;
            end
            
            updatedStructure = cell2mat(updatedStructure);
        end
        
    end
    
    properties (Access = private, Hidden = true)
        
        %Object properties (for internal use)
        
        ciValues    = [];
        csValues    = [];
        
        outDir   = [];
        imageDir = [];
        fileDir  = [];
        
        labelIDs  = [];
        labelText = {};
        
        spectrogramDimM = [];
        spectrogramDimN = [];
    end
    
    properties (SetAccess = private, Hidden = false)
        
        % Object properties, treat as constant struct fields.
        % objects can be used to retrieve output some saved instances. 
        % ie: 
        
        complete            = false;
        spectralFeatures    = [];
        classStructureArray = [];
        structureIn = [];
        
    end
    
end

