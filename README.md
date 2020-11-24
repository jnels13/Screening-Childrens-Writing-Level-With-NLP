# Applying Machine Learning to Assess Student Writing Level

This model is a proof-of-concept machine-learning model to assess student writing level based upon a corpus of previously categorized texts. Given the size of the available corpus and that some texts had been attributed to multiple grades, texts were grouped into the following grade-level categories: early elementary (k-2), middle-elementary (3-4), middle school (5-8) and high school (9-12). 

The model is not a "grammar checker" or equivalent, and it is subject-matter agnostic. It exclusively classifies a given text based upon a corpus of previously graded materials. 

### About the dataset:

Three datasets are used and referenced herein: 
<br>> writingcsv.csv is the first corpus of texts from multiple sources (n=188)
<br>> writingcsv2.csv is the single-source corpus targeted to meet the Common Core standards (n=100)
<br>> writingcsv2_combined is the combined corpus (n=288)

Writing samples in the dataset were obtained from several different sources and comprise many different types. Some sources had corrected obvious misspellings, so obvious misspellings were corrected in the remaining texts for consistency. Bibliographies from research papers were not included.  Sources varied, though the vast majority of texts were obtained from https://achievethecore.org (comprising the second set, n=100), https://k12.thoughtfullearning.com, and http://www.ttms.org. Licensure and copyright information for the texts may be viewed on each site. 

###  Initial Review of the Corpus

Initial EDA revealed that the average number of words per text increased as grade level increased. Middle-school and high-school distributions were very close, as shown below, though high-school texts had broader dispersion, likely due to the different types of texts. 

<img src="https://github.com/jnels13/Screening-Childrens-Writing-Level-With-NLP/blob/main/Source%20Images/3_numwords.png" width="592" height="413">

Similarly, average word length per text was correlated with increase in grade level, in all three data-set groups. The differences were negligible, though less so in the second, single-source group.  After lemmatization and stop-word removal, the differences between the top two groupings (middle and high school) are nearly gone, though the differences between the lower grades remain largely the same.

**Group 3 (Combination of Previous Two):** 
<img src="https://github.com/jnels13/Screening-Childrens-Writing-Level-With-NLP/blob/main/Source%20Images/3_wordlen.png">

Further, the actual words used across the different grade-groups may be viewed in a word cloud: 

<img src="https://github.com/jnels13/Screening-Childrens-Writing-Level-With-NLP/blob/main/Source%20Images/wc_0.png" width="370" height="258"> <img src="https://github.com/jnels13/Screening-Childrens-Writing-Level-With-NLP/blob/main/Source%20Images/wc_3.png" width="370" height="258">

<img src="https://github.com/jnels13/Screening-Childrens-Writing-Level-With-NLP/blob/main/Source%20Images/wc_5.png" width="370" height="258"> <img src="https://github.com/jnels13/Screening-Childrens-Writing-Level-With-NLP/blob/main/Source%20Images/wc_9.png" width="370" height="258">

Themes repeat across grade-groups, such as "dog" and "cat" decreassing in frequency after the early grades, "time" growing in frequency through the different grade groups, and "mom" becoming "mother" in high school.  The repetition of words across grade groups from the youngest to the eldest students is also interesting. 

### Model Development

#### Preprocessing

Standard preprocessing included tokenization of the texts, removal of English stop words and punctuation, and lemmatization. 

#### Model Selection

Three models were used: support-vector machine, random forest, and XG Boost. All models used RandomizedSearchCV to tune the parameters (parameter grids were further tweaked in edge cases), and a dummy classifier was used to evaluate the modes' performance over baseline. The models were run on each of the three "groups" of data to examine the efficacy of the models as the corpus size increased (from n=100, to n=188, to n=288). The data was applied to each model utilizing two strategies for categorizing the data: TF-IDF weighting and Word2vec vectorization.  SMOTE was used to account for class imbalance. 

### Results

Some of the initial models improved overall as the corpus size increased, as shown by the F1 scores below:

<img src="https://github.com/jnels13/Screening-Childrens-Writing-Level-With-NLP/blob/main/Source%20Images/F1_Scores.png">

The best-performing model was the Random Forest using TF-IDF weights (63.218% accuracy, 58.674 F1 score), and it clearly improved in performance as corpus size increased. It can correctly classify texts roughly twice as good as the baseline (32.184% accuracy). The best Word2vec model was XGBoost, with 50.575% accuracy and a 50.282 F1 score.

Texts may be pasted directly into the notebook, though this will soon be transferred to a stand-alone app.  Further work includes updating the corpus size as additional texts become available, and deploying to a stand-alone app. The source data should also be checked for potential bias in its origin, and thus, in its application; that goes beyond the scope of this initial developed model.

### Further Reading

I found the paper linked below while searching for a "ready made" data set. It generally asks the same question as I do, though the dataset was not obtainable (the English source material was no longer available at the referenced web site). The English data appears also to be sourced from a variety of English-speaking countries outside of the U.S. and the samples I could obtain from the Internet Archive were very short.  My model may differ as it appears that the source texts are primarily from the United States.  https://github.com/sgjimenezv/children_age_narrative_dataset/blob/master/PAPER_CICLING_2014_Moreno_Jimenez_Baquero_pre-print.pdf
