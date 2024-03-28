import numpy as np
import pandas as pd

def evaluate_overall(instances, human_feature_names, model_rationale_names):

    print(','+'n,'+','.join(model_rationale_names))
    for human_feature_name in human_feature_names:
        mean_correlations = []
        for model_rationale_name in model_rationale_names:
            column = human_feature_name+'-VS-'+model_rationale_name
            mean_corr = instances[column].mean()     
            mean_correlations.append(str(round(mean_corr, 3)))
        print(human_feature_name+','+str(instances[column].count())+','+','.join(mean_correlations))


def evaluate_per_category(cat_name, instances, human_feature_names, model_rationale_names):

    unique_cats = instances[cat_name].unique()
    header_column1 = [name for name in model_rationale_names for cat in unique_cats]
    header_column2 = [str(cat) for name in model_rationale_names for cat in unique_cats]
    print(','+'n,'*len(unique_cats)+','.join(header_column1))
    print(','+','.join([str(cat) for cat in unique_cats])+','+','.join(header_column2))
    for human_feature_name in human_feature_names:
        mean_correlations = []
        for model_rationale_name in model_rationale_names: #TEST
            column = human_feature_name+'-VS-'+model_rationale_name
            mean_corr = instances.groupby(cat_name)[column].mean()
            cat_corrs = [str(round(mean_corr[cat], 3)) for cat in unique_cats]
            mean_correlations.append(cat_corrs)
        n = [str(instances.groupby(cat_name)[column].count()[cat]) for cat in unique_cats]
        print(human_feature_name+','+','.join(n)+','+','.join([','.join(m) for m in mean_correlations]))

def main(filepath, human_feature_names, model_rationale_names, category_names=None):

    instances = pd.read_csv(filepath)

    evaluate_overall(instances, human_feature_names, model_rationale_names)

    if category_names:
        for cat_name in category_names:
            print()
            evaluate_per_category(cat_name, instances, human_feature_names, model_rationale_names)
            

if __name__ == '__main__':
    
    filepath = '../data/all_output.csv'
    human_features = ['IA_FIXATION_%', 'IA_RUN_COUNT', 'IA_DWELL_TIME_%',
                    'IA_AVERAGE_FIX_PUPIL_SIZE', 'IA_MAX_FIX_PUPIL_SIZE', 
                    'IA_MIN_FIX_PUPIL_SIZE', 'Pupilsize_variation',
                    'IA_FIRST_FIXATION_DURATION', 'IA_FIRST_RUN_FIXATION_%',
                    'IA_REGRESSION_IN_COUNT', 'IA_REGRESSION_OUT_COUNT',
                    'backward_reg_count', 'forward_reg_count', 'total_reg_count',
                    'IA_REGRESSION_IN', 'IA_REGRESSION_OUT', 'IA_SKIP',
                    'Clicked']
    #model_rationales = ['input_x_gradient', 'saliency', 'shapley_value']
    #category_names = ['Intensity_Category_Binary']
    model_rationales = ['input_x_gradient']
    category_names = ['RECORDING_SESSION_LABEL']
    main(filepath, human_features, model_rationales, category_names=category_names)