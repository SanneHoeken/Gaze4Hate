import pandas as pd
from sklearn.metrics import classification_report

def evaluate_overall(model_names, allmodel_preds, gold):

    overall_reports = []
    for i in range(len(model_names)):
        overall_reports.append(classification_report(gold, allmodel_preds[i], output_dict=True))

    print(',n,', ','.join(model_names))
    print(f"HATE,{overall_reports[0]['1']['support']},", ','.join([str(round(overall_reports[i]['1']['f1-score'], 2)) for i in range(len(overall_reports))]))
    print(f"NO HATE,{overall_reports[0]['0']['support']},", ','.join([str(round(overall_reports[i]['0']['f1-score'], 2)) for i in range(len(overall_reports))]))
    print(f"acc.,{overall_reports[0]['macro avg']['support']},", ','.join([str(round(overall_reports[i]['accuracy'], 2)) for i in range(len(overall_reports))]))
    print(f"macro avg,{overall_reports[0]['macro avg']['support']},", ','.join([str(round(overall_reports[i]['macro avg']['f1-score'], 2)) for i in range(len(overall_reports))]))
    print(f"weighted avg,{overall_reports[0]['weighted avg']['support']},", ','.join([str(round(overall_reports[i]['weighted avg']['f1-score'], 2)) for i in range(len(overall_reports))]))
    

def evaluate_splitpercategory(model_names, allmodel_preds, gold, cats):

    unique_cats = set(cats)
    cat_reports = {cat: [] for cat in unique_cats}

    for cat in unique_cats:
        for i in range(len(model_names)):
            cat_preds = [p for p, c in zip(allmodel_preds[i], cats) if c == cat]
            cat_gold = [g for g, c in zip(gold, cats) if c == cat]
            cat_reports[cat].append(classification_report(cat_gold, cat_preds, output_dict=True))

    print(',,n,', ','.join(model_names))
    for cat in unique_cats:
        print(f"HATE,{cat},{cat_reports[cat][0]['1']['support']},", ','.join([str(round(cat_reports[cat][i]['1']['f1-score'], 2)) for i in range(len(cat_reports[cat]))]))
    for cat in unique_cats:
        print(f"NO HATE,{cat},{cat_reports[cat][0]['0']['support']},", ','.join([str(round(cat_reports[cat][i]['0']['f1-score'], 2)) for i in range(len(cat_reports[cat]))]))
    for cat in unique_cats:
        print(f"'macro avg',{cat},{cat_reports[cat][0]['macro avg']['support']},", ','.join([str(round(cat_reports[cat][i]['macro avg']['f1-score'], 2)) for i in range(len(cat_reports[cat]))]))

def main(filepath, model_names, category_names=None):

    instances = pd.read_csv(filepath)
    allmodel_preds = [instances[model_name+'_prediction'] for model_name in model_names]
    gold = instances['Intensity_Category_Binary']

    evaluate_overall(model_names, allmodel_preds, gold)

    if category_names:
        for cat_name in category_names:
            cats = instances[cat_name]
            print()
            evaluate_splitpercategory(model_names, allmodel_preds, gold, cats)

if __name__ == '__main__':

    preds_filepath = '../data/all_output.csv'
    #model_names = ['deepset', 'ortiz', 'aluru', 'rott', 'ml6', 'rott-hc']
    #category_names=['Ling_type', 'split']

    model_names = ['rott-hc']
    category_names = ['RECORDING_SESSION_LABEL']
    main(preds_filepath, model_names, category_names=category_names)