%% User friendly LDA in Matlab
%  This script is derivative of a basic LDA topic model
%  published by MathWorks implemented in Matlab R2020b.
%  Details are publically available here:
%   https://www.mathworks.com/help/textanalytics/ug/analyze-text-data-using-topic-models.html
%   including the preprocessText function.

%% 
% Prep data
% Read in documents
data=readtable("Documents.csv",'TextType','string');

% Analyze questions separately
textData = data.Q5;
textData(1:5);

% Preprocess
documents = preprocessText(textData);
%% 
% Choose n topics
% Set aside 50% for validation
% Iterate 10 times, inspect charts.
numDocuments = numel(documents);
cvp = cvpartition(numDocuments,'HoldOut',0.5);
documentsTrain = documents(cvp.training);
documentsValidation = documents(cvp.test);

% Make a bag 'o words.
bag = bagOfWords(documentsTrain);
bag = removeInfrequentWords(bag,2);
bag = removeEmptyDocuments(bag);

% Choose number of topics
numTopicsRange = [5 10 15 20 40 50 60 70 80 90 100 150 200 250 300];
for i = 1:numel(numTopicsRange)
    numTopics = numTopicsRange(i);
    
    mdl = fitlda(bag,numTopics, ...
        'Solver','savb', ...
        'Verbose',0);
    
    [~,validationPerplexity(i)] = logp(mdl,documentsValidation);
    timeElapsed(i) = mdl.FitInfo.History.TimeSinceStart(end);
end

figure
yyaxis left
plot(numTopicsRange,validationPerplexity,'+-')
ylabel("Validation Perplexity")

yyaxis right
plot(numTopicsRange,timeElapsed,'o-')
ylabel("Time Elapsed (s)")

legend(["Validation Perplexity" "Time Elapsed (s)"],'Location','southeast')
xlabel("Number of Topics")

%%
% Fit LDA: 

% First overwrite bag of words for the whole set of documents.
bag = bagOfWords(documents);
bag = removeInfrequentWords(bag,2);
bag = removeEmptyDocuments(bag);

numTopics = 150;
mdl = fitlda(bag,numTopics,'Verbose',0,...
               'FitTopicProbabilities',true,...
               'TopicOrder','initial-fit-probability');
           
figure
bar(mdl.CorpusTopicProbabilities)
ylabel("Probability")
xlabel("Topic Number")
title("Corpus Topic Probabilities")
xline(10.5)

% What is the total probability of the most likely topics?
sum(mdl.CorpusTopicProbabilities(1:10)) % 5 for Q3,4 and 10 for Q5

% Wordclouds and barplots
figure;
k=5;
for topicIdx = 1:10
    subplot(2,5,topicIdx);
%     tbl = topkwords(mdl,k,topicIdx);
%     barh(tbl.Score);
%     yticklabels(tbl.Word);
    wordcloud(mdl,topicIdx);
    title("Topic " + topicIdx);
end




%% Explore the topic content by phrases you care about.
newDocument = tokenizedDocument("cognitive");
topicMixture = transform(mdl,newDocument);
figure
bar(topicMixture)
xlabel("Topic Index")
ylabel("Probability")
title("Document Topic Probabilities")

%% More exploration
myTopic = 12; %Which topic do you want to find representative responses for?

disp(mdl.CorpusTopicProbabilities(myTopic));

topicMatch=zeros(length(documents),3);
docScores=zeros(length(documents),numTopics); 
for docIdx=1:length(documents)
    docScores=transform(mdl,documents(docIdx));
    [~,~,~,AUC]=perfcurve(1:numTopics,docScores,myTopic);
    topicMatch(docIdx,:)=[docIdx,...
        docScores(myTopic),AUC]; %Get the AUC for a document.
end
topicMatch=sortrows(topicMatch,[3 2],'descend'); %sorted by AUC and Probability
textData(topicMatch(1:10,1))


%% Visualize topic mixtures.
figure
topicMixtures = transform(mdl,documents(1:5));
barh(topicMixtures(1:5,:),'stacked')
xlim([0 1])
title("Topic Mixtures")
xlabel("Topic Probability")
ylabel("Document")
legend("Topic " + string(1:numTopics),'Location','northeastoutside')
