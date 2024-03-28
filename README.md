# Gaze4Hate

Repository for the paper: Özge Alaçam, Sanne Hoeken and Sina Zarrieß. 2024. Eyes Don't Lie: Subjective Hate Annotation and Detection with Gaze. [Manuscript submitted for publication].

The paper contributes a new dataset GAZE4HATE that provides gaze and annotation data from hate speech annotators. **/data** contains a sample of the dataset, the full dataset can be made available upon request. 

The paper studies whether the gaze of an annotator does provide predictors of her subjective hatefulness rating, and whether gaze features can be used to evaluate and improve hate speech detection models. 

The analyses and experiments are three-fold:
1. **/code/initial-data-analysis** contains all code for the statistical modeling on our collected eye-tracking and annotation data 
2. **/code/hate-speech-detection** contains all code for the evaluation of a range of existing hate speech detection models on our data, comparing models' and humans' rationales to human gaze 
3. **/code/meanion_model** contains all code for the MEANION model, which integrates text-based hate speech detection with gaze features.