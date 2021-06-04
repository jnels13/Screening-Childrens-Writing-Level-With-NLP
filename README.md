<img align="center" src="https://github.com/jnels13/Screening-Childrens-Writing-Level-With-NLP/blob/main/Source%20Images/StudentWritingBanner.JPG" width="auto" height="auto">

Try the final model on Streamlit <a href="https://share.streamlit.io/jnels13/screening-childrens-writing-level-with-nlp/main/Mod4_Final_Streamlit/student_text_streamlit.py"> HERE </a>.  (Streamlit puts shared apps to bed after seven days, so you may have to wait a moment for it to load.)

# Applying Machine Learning to Assess Student Writing Level

This model is a proof-of-concept machine-learning model to assess student-writing level based upon a corpus of previously categorized texts. Given the size of the available corpus and that some texts had been attributed to multiple grades, texts were grouped into the following grade-level categories: early elementary (k-2), middle-elementary (3-4), middle school (5-8) and high school (9-12). 

The model is not a "grammar checker," and it is subject-matter agnostic. It only classifies a given text based upon a corpus of previously graded materials. The repository includes the notebook, the datasets described below, a high-level overview of the project (the PDF document), and the source images referenced herein. 

### About the dataset:

The dataset, Combined.csv (n=289), is a combination from two sources. Writing samples in the dataset were obtained from several different sources and comprise many different types. Some sources had corrected obvious misspellings, so obvious misspellings were corrected in the remaining texts for consistency. Bibliographies from research papers were not included.  Sources varied, though the vast majority of texts were obtained from https://achievethecore.org (comprising the second set, n=100), https://k12.thoughtfullearning.com, and http://www.ttms.org. Licensure and copyright information for the texts may be viewed on each site. 

###  Initial Review of the Corpus

Initial EDA revealed that both the average number of words per text and the average word length increased as grade level increased. 

As shown below, middle-school and high-school distributions were very close, though high-school texts had broader dispersion, likely due to the different types of texts:

<img src="https://github.com/jnels13/Screening-Childrens-Writing-Level-With-NLP/blob/main/Source%20Images/3_numwords.png" width="592" height="413">

Similarly, average word length per text was correlated with increase in grade level, though the differences were negligible.  After lemmatization and stop-word removal, the differences between the top two groupings (middle and high school) are nearly gone, though the differences between the lower grades remain largely the same: 

<img src="https://github.com/jnels13/Screening-Childrens-Writing-Level-With-NLP/blob/main/Source%20Images/3_wordlen.png">

The actual words used across the different grade-groups may be viewed in a word cloud: 

<img src="https://github.com/jnels13/Screening-Childrens-Writing-Level-With-NLP/blob/main/Source%20Images/wc_3.png">

The ten most frequent words per grade-group are visualized below: 
<img src="https://github.com/jnels13/Screening-Childrens-Writing-Level-With-NLP/blob/main/Source%20Images/most_important.png">

When reviewing the following two preceding visualizations, themes repeat across grade-groups, such as "dog" and "cat" decreasing in frequency after the early grades, "time" growing in frequency through the different grade groups, and "mom" becoming "mother" in high school.  The repetition of words across grade groups from the youngest to the eldest students is also interesting. 

### Model Development

#### Preprocessing

Standard preprocessing included tokenization of the texts, removal of English stop words and punctuation, and lemmatization. 

#### Model Selection

Four models were used: logistic regression, support-vector machine, random forest, and XG Boost. All models used RandomizedSearchCV to tune the parameters (parameter grids were further tweaked in edge cases), and a dummy classifier was used to evaluate the modes' performance over baseline. The data was applied to each model utilizing two strategies for categorizing the data: TF-IDF weighting and Word2vec vectorization.  SMOTE was used to account for class imbalance. 

### Results

The best-performing model was the Random Forest using TF-IDF weights (64.368% Accuracy, 60.142 F1 score). It correctly classified texts roughly three times as well as the baseline (20.690% accuracy).  The final app is located <a href="https://share.streamlit.io/jnels13/screening-childrens-writing-level-with-nlp/main/Mod4_Final_Streamlit/student_text_streamlit.py"> HERE </a>.  

### Future Work

Future work includes growing the corpus, potentially classifying text using both the tf-idf weights AND the dense, traditional factors such as word and text length.  The source data should also be checked for potential bias in its origin, and thus, in its application; that goes beyond the scope of this initial developed model.

### Further Reading

I found the paper linked below while searching for a "ready made" data set. It generally asks the same question as I do, though the dataset was not obtainable (the English source material was no longer available at the referenced web site). The English data appears also to be sourced from a variety of English-speaking countries outside of the U.S. and the samples I could obtain from the Internet Archive were very short.  My model may differ as it appears that the source texts are primarily from the United States.  https://github.com/sgjimenezv/children_age_narrative_dataset/blob/master/PAPER_CICLING_2014_Moreno_Jimenez_Baquero_pre-print.pdf
