import torch, json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


def predict(model, encodings):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    masks = torch.tensor([[int(e > 0) for e in encodings]], device=device)
    input = torch.tensor([encodings], dtype=torch.long, device=device)
    
    # get model output
    output = model(input, masks)
    pred_id = torch.argmax(output[0], dim=1).item()
    #pred_label = model.config.id2label[pred_id]

    return pred_id


def main(test_filepath, output_filepath, models):

    # get input
    with open(test_filepath, 'r') as infile:
        sno2instances = json.load(infile)
    instances = pd.DataFrame(list(sno2instances.values()))
    
    # encode input
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model_name, dir in models.items():

        tokenizer = AutoTokenizer.from_pretrained(dir['tokenizer_dir'])
        encodings = [tokenizer.encode(t) for t in instances['assertion']]
        
        # predict testset
        model = AutoModelForSequenceClassification.from_pretrained(dir['model_dir']).to(device)
        predictions = [predict(model, e) for e in tqdm(encodings)]
        if any([p > 1 for p in predictions]):
            print('converting non-binary predictions to binary...')
            predictions = [0 if p == 0 else 1 for p in predictions]
        
        instances[model_name+'_prediction'] = predictions
    
    # save output
    #instances.to_csv(output_filepath.replace('json', 'csv'), index=False) 
    for sno, inst in zip(sno2instances.keys(), instances.to_dict('records')):
        sno2instances[sno] = inst

    with open(output_filepath, 'w') as outfile:
        json.dump(sno2instances, outfile)
    


if __name__ == '__main__':

    test_filepath = '../data/data_per_participant.json'
    models = {'deepset': {'model_dir': 'deepset/bert-base-german-cased-hatespeech-GermEval18Coarse', 'tokenizer_dir': 'deepset/bert-base-german-cased-hatespeech-GermEval18Coarse'},
              'ortiz': {'model_dir': 'jorgeortizv/BERT-hateSpeechRecognition-German', 'tokenizer_dir': 'jorgeortizv/BERT-hateSpeechRecognition-German'},
              'aluru': {'model_dir': 'Hate-speech-CNERG/dehatebert-mono-german', 'tokenizer_dir': 'Hate-speech-CNERG/dehatebert-mono-german'},
              'rott': {'model_dir': 'chrisrtt/gbert-multi-class-german-hate', 'tokenizer_dir': 'chrisrtt/gbert-multi-class-german-hate'},
              'ml6': {'model_dir': 'ml6team/distilbert-base-german-cased-toxic-comments', 'tokenizer_dir': 'ml6team/distilbert-base-german-cased-toxic-comments'},
              'rott-hc': {'model_dir': '../finetuned_models/rott-hatecheck', 'tokenizer_dir': 'chrisrtt/gbert-multi-class-german-hate'}}
    output_filepath = '../data/model-predictions.json'
    
    main(test_filepath, output_filepath, models)