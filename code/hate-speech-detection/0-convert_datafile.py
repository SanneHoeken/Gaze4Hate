import json, operator
import numpy as np
import pandas as pd

def per_participant_json(all_data_filename, output_filename, sent_params, sent_ptcp_params, token_ptcp_params):

    all_data_df = pd.read_csv(all_data_filename, sep='\t') 
    all_data_df['sno'] = all_data_df['sno'].astype(str)
    all_data_df = all_data_df.sort_values(by=['RECORDING_SESSION_LABEL', 'sno'])
    sno2instances = {sno: dict() for sno in all_data_df['sno']}

    for sno, instance in sno2instances.items():
        for sent_param in sent_params:
            instance[sent_param] = all_data_df.loc[all_data_df['sno'] == sno, sent_param].iloc[0]
        
        for param in token_ptcp_params + sent_ptcp_params:
            instance[param] = []
    
    current_sno = None
    
    for _, row in all_data_df.iterrows():

        sno = row['sno']
        if sno != current_sno:
            
            if current_sno != None:
                for param, param_value in zip(sent_ptcp_params, sno_param_values):
                    sno2instances[current_sno][param].append(param_value)
                
                for n, param in enumerate(token_ptcp_params):                
                    sno2instances[current_sno][param].append(token_param_lists[n])

            current_sno = sno
            sno_param_values = [row[param] for param in sent_ptcp_params]
            token_param_lists = [[] for _ in range(len(token_ptcp_params))]
        
        for n, param_list in enumerate(token_param_lists):
            if token_ptcp_params[n] == 'Clicked':
                param_list.append(0 if row['Clicked'] == False else 1)
            else:
                param_list.append(float(row[token_ptcp_params[n]]))

    for param in sent_ptcp_params:
        sno2instances[current_sno][param].append(row[param])

    for n, param in enumerate(token_ptcp_params):                
        sno2instances[current_sno][param].append(token_param_lists[n])

    with open(output_filename, 'w') as outfile:
        json.dump(sno2instances, outfile)
    

"""
def participant_avg_json(perparticipant_filename, output_filename, sent_params, token_ptcp_params, label_column, unique_labels):

    with open(perparticipant_filename, 'r') as infile:
        sno2instances = json.load(infile)

    sno2avginstances = {}

    for sno, dic in sno2instances.items():

        new_dic = {sent_param: dic[sent_param] for sent_param in sent_params}
        new_dic['label2freq'] = {label: 0 for label in unique_labels}
        for param in token_ptcp_params:
            new_dic['avg_'+param]= {label: [] for label in unique_labels}

        for i, label in enumerate(dic[label_column]):
            new_dic['label2freq'][label] += 1
            for param in token_ptcp_params:
                new_dic['avg_'+param][label].append(dic[param][i])

        for label in unique_labels:
            if new_dic['label2freq'][label] > 0:
                for param in token_ptcp_params:
                    new_dic['avg_'+param][label] = np.mean(new_dic['avg_'+param][label], axis=0).tolist()

        new_dic['majority_label'] = max(new_dic['label2freq'].items(), key=operator.itemgetter(1))[0]
        sno2avginstances[sno] = new_dic

    with open(output_filename, 'w') as outfile:
        json.dump(sno2avginstances, outfile)
"""

if __name__ == '__main__':

    all_data_filename = ''
    output_filename = '../data/data_per_participant.json'
    sent_params = ['sno', 'assertion', 'Ling_type', 'split']
    sent_ptcp_params = ['RECORDING_SESSION_LABEL', 'Intensity_Category', 'Intensity_Category_Binary']
    token_ptcp_params = ['IA_FIXATION_%', 'IA_RUN_COUNT', 'IA_DWELL_TIME_%',
                    'IA_AVERAGE_FIX_PUPIL_SIZE', 'IA_MAX_FIX_PUPIL_SIZE', 
                    'IA_MIN_FIX_PUPIL_SIZE', 'Pupilsize_variation',
                    'IA_FIRST_FIXATION_DURATION', 'IA_FIRST_RUN_FIXATION_%',
                    'IA_REGRESSION_IN_COUNT', 'IA_REGRESSION_OUT_COUNT',
                    'backward_reg_count', 'forward_reg_count', 'total_reg_count',
                    'IA_REGRESSION_IN', 'IA_REGRESSION_OUT', 'IA_SKIP',
                    'Clicked']

    per_participant_json(all_data_filename, output_filename, sent_params, sent_ptcp_params, token_ptcp_params)
    