# Applying Machine Learning to Assess Student Writing Level

This model is being offered as a proof-of-concept deep-learning model to assess student writing level based upon a corpus of previously categorized texts. Given the size of the available corpus and that some texts had been attributed to multiple grades, texts were grouped into the following grade-level categories: early elementary (k-2), middle-elementary (3-4), middle school (5-8) and high school (9-12). 

### About the dataset:

Writing samples in the dataset were obtained from several different sources and comprise many different types (poetry, narrative, etc.). Some sources had corrected obvious misspellings, so obvious misspellings were corrected in the remaining texts for consistency. Bibliographies from research papers were not included.  Sources varied, though the vast majority of texts were obtained from https://achievethecore.org, https://k12.thoughtfullearning.com, and http://www.ttms.org. Licensure and copyright information for the texts may be viewed on each site. 

###  Initial Review of the Corpus

Initial EDA revealed that the average number of words per text increased as grade level increased, though middle-school and high-school distributions were very close, as shown below, though high-school texts had broader dispersion, likely due to the different types of texts (i.e., poetry and research papers). 

[VISUALIZATION 1]

Similarly, average word length per text was correlated with increase in grade level, though the differences were negligible: 

[VISUALIZATION 2]

After lemmatization and stop-word removal, the differences between the top two groupings (middle and high school) are nearly gone, though the differences between the lower grades remain largely the same:

[VISUALIZATION 3]

### Model Development

#### Preprocessing

Preprocessing began with removal of the English stop words and punctuation. 


#### Model Selection



### Further Reading

I found the paper linked below while searching for a "ready made" data set. It asks the same question as I do, though the dataset was not obtainable (the English source material was no longer available at the referenced web site). The English data appears also to be sourced from a variety of English-speaking countries outside of the U.S.; my model may differ as it appears that the source texts are primarily from the United States.  https://github.com/sgjimenezv/children_age_narrative_dataset/blob/master/PAPER_CICLING_2014_Moreno_Jimenez_Baquero_pre-print.pdf

# Screening-Childrens-Writing-Level-With-NLP
