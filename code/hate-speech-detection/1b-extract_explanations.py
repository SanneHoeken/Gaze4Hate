from captum.attr import ShapleyValueSampling, InputXGradient, Saliency
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, json
from tqdm import tqdm
import pandas as pd

# inspired by: XAI Benchmark (https://github.com/copenlu/xai-benchmark) and Captum Tutorials (https://captum.ai/tutorials/)
# this code only works for BERT classification models

class ShapleyModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ShapleyModelWrapper, self).__init__()
        self.model = model

    def forward(self, input, mask):
        return self.model(input, attention_mask=mask)[0]

class GradientModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(GradientModelWrapper, self).__init__()
        self.model = model

    def forward(self, input, mask):
        return self.model(inputs_embeds=input, attention_mask=mask)[0]


def explain_input(encodings, classes, model, ablator_name, ablator, device):
    
    # prepare inputs
    masks = torch.tensor([[int(i > 0) for i in encodings]], device=device)
    input = torch.tensor([encodings], dtype=torch.long, device=device)
      
    if ablator_name != 'shapley_value':  
      input = model.model.bert.embeddings(input)

    # get attributions for every class
    cls2attributions = dict()
    for cls in classes:
        attributions = ablator.attribute(input, target=cls, additional_forward_args=masks)[0]
        if ablator_name != 'shapley_value':
            # l2 summarization
            attributions = attributions.norm(p=1, dim=-1).squeeze(0)
            
        cls2attributions[cls] = attributions.tolist()
    
    return cls2attributions


def main(input_filepath, output_filepath, model_dir, tokenizer_dir):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get input
    with open(input_filepath, 'r') as infile:
        sno2instances = json.load(infile)
    instances = pd.DataFrame(list(sno2instances.values()))

    # prepare tokenizer and encode input
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    encodings = [tokenizer.encode(t) for t in instances['assertion']]
    instances['subwords'] = [tokenizer.convert_ids_to_tokens(e) for e in encodings]
    
    # compute attributions    
    for ablator_name in ['input_x_gradient', 'saliency', 'shapley_value']:
        
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
        classes = list(model.config.id2label.keys())

        if ablator_name == 'shapley_value':
            model = ShapleyModelWrapper(model)
            ablator = ShapleyValueSampling(model)
        else:
            model = GradientModelWrapper(model)
            if ablator_name == 'input_x_gradient':
                ablator = InputXGradient(model)
            elif ablator_name == 'saliency':
                ablator = Saliency(model)

        cls2attributions = []
        for encoding in tqdm(encodings):
            cls2attributions.append(explain_input(encoding, classes, model, ablator_name, ablator, device))
        
        instances[ablator_name] = cls2attributions

    for sno, inst in zip(sno2instances.keys(), instances.to_dict('records')):
        sno2instances[sno] = inst

    with open(output_filepath, 'w') as outfile:
        json.dump(sno2instances, outfile)


if __name__ == '__main__':

    input_filepath = '../data/model-predictions.json'
    output_filepath = '../data/rott-hc-explanation_values.json'
    model_dir = '../finetuned_models/rott-hatecheck'
    tokenizer_dir = 'chrisrtt/gbert-multi-class-german-hate'
    
    main(input_filepath, output_filepath, model_dir, tokenizer_dir)