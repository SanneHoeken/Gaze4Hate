import numpy as np
import json
from scipy.stats import pearsonr
import pandas as pd

def subword_to_gold_level(subword_explanation, subwords):
    
    # REALLY DATASET SPECIFIC AND SIMPLISTIC RULE-BASED METHOD
    
    gold_level_explanation = [] 
    word_exps = []

    for i, (subword, exp) in enumerate(zip(subwords, subword_explanation)):

        if subword in ['[CLS]', '[SEP]', '.', ',']:
            continue
        else:
            word_exps.append(exp)
            if not any([subwords[i+1].startswith('##'),
                        subword in ['#', '-'],
                        subwords[i+1] == '-', 
                        subwords[i-1] == '#']):
                gold_level_explanation.append(np.mean(word_exps))
                word_exps = []
            
    return gold_level_explanation


def main(filepath, output_filepath, human_feature_names, model_rationale_names=[]):

    with open(filepath, 'r') as infile:
        sno2instances = json.load(infile)

    instances = []

    for inst in sno2instances.values():
        for i in range(len(inst['Intensity_Category_Binary'])):
            instance = dict()
            for key, value in inst.items():
                if key not in human_feature_names + model_rationale_names + ['subwords']:
                    if type(value) != list:
                        instance[key] = value
                    else: 
                        instance[key] = value[i]

            if model_rationale_names:        
                for human_feature_name in human_feature_names:
                    for model_rationale_name in model_rationale_names:
                        cls = inst['Intensity_Category_Binary'][i]
                        human_feat = inst[human_feature_name][i]
                        model_rat = subword_to_gold_level(inst[model_rationale_name][str(cls)], inst['subwords'])
                        r, p = pearsonr(human_feat, model_rat) # is this a proper way for evaluation?
                        instance[human_feature_name+'-VS-'+model_rationale_name] = r

            instances.append(instance)

    all_data = pd.DataFrame(instances)
    all_data.to_csv(output_filepath)

    

if __name__ == '__main__':
    
    filepath = '../data/rott-hc-explanation_values.json'
    output_filepath = '../data/all_output.csv'
    human_features = ['IA_FIXATION_%', 'IA_RUN_COUNT', 'IA_DWELL_TIME_%',
                    'IA_AVERAGE_FIX_PUPIL_SIZE', 'IA_MAX_FIX_PUPIL_SIZE', 
                    'IA_MIN_FIX_PUPIL_SIZE', 'Pupilsize_variation',
                    'IA_FIRST_FIXATION_DURATION', 'IA_FIRST_RUN_FIXATION_%',
                    'IA_REGRESSION_IN_COUNT', 'IA_REGRESSION_OUT_COUNT',
                    'backward_reg_count', 'forward_reg_count', 'total_reg_count',
                    'IA_REGRESSION_IN', 'IA_REGRESSION_OUT', 'IA_SKIP',
                    'Clicked']
    model_rationales = ['input_x_gradient', 'saliency', 'shapley_value']
    
    main(filepath, output_filepath, human_features, model_rationales=model_rationales)