import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig, pipeline, AutoModelForCausalLM
import torch
from torch import cuda
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
import cohere
import numpy as np
from sklearn.preprocessing import normalize

def concatenate_saved_embeddings(dataset_1, dataset_2):

    embeddings_1 = dataset_1.embedding.tolist()
    embeddings_2 = dataset_2.embedding.tolist()

    #normalize embeddings
    embeddings_1 = normalize(np.array(embeddings_1))
    embeddings_2 = normalize(np.array(embeddings_2))

    concatenated_embeddings = [np.concatenate((e1, e2), axis=None) for e1, e2 in zip(embeddings_1, embeddings_2)]
    concatenated_embeddings = np.array(concatenated_embeddings)

    X_embedded = TSNE(n_components=2).fit_transform(concatenated_embeddings)
    concatenated_embeddings = concatenated_embeddings.tolist()

    results = pd.DataFrame({
        dataset_1.content.name: dataset_1.content,
        dataset_1.label.name: dataset_1.label,
        dataset_1.pred.name: dataset_1.pred,
        dataset_1.string_label.name: dataset_1.string_label,
        dataset_1.string_pred.name: dataset_1.string_pred,
        dataset_1.loss.name: dataset_1.loss,
        'x': X_embedded[:,0],
        'y': X_embedded[:,1],
        'embedding': concatenated_embeddings,
    })

    return results

def get_model_and_tokenizer(embedding_model):
    #load the model and tokenizer on GPU if available
    device = 'cuda' if cuda.is_available() else 'cpu'

    #get the tokenizer and model. we're using Auto so that any model can be used
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    model = AutoModelForSequenceClassification.from_pretrained(embedding_model).to(device)  

    #get the hidden states    
    model.config.output_hidden_states = True
    model.config.output_attentions = True

    return model, tokenizer

def last_hidden_state(embedding_model, dataset):

    model, tokenizer = get_model_and_tokenizer(embedding_model)

    #model to cuda
    device = 'cuda' if cuda.is_available() else 'cpu'

    #use DataLoader to parallelise the inference
    dataloader = DataLoader(dataset, batch_size = 4, num_workers = 0)

    #cross entropy for loss function
    criterion = nn.CrossEntropyLoss(reduction='none')
    softmax = nn.Softmax(dim=1)

    #model in evaluation mode
    model.eval()

    #empty tensors
    labels_tensor = torch.Tensor().to(device) #save actual label
    targets = torch.Tensor().to(device) 
    embeddings =  torch.Tensor().to(device) #save the last hidden state
    losses = torch.Tensor().to(device) #save the loss
    outputs_tensor = torch.Tensor().to(device) #save the predictions

    text = []

    #disable gradient calculation (not necessary for inference)
    with torch.no_grad():

        #iterate over the bateches
        for i_batch, batch in tqdm(enumerate(dataloader), total = len(dataloader), position = 0, leave = True):

            #get the examples from the current batch
            examples = batch['text']

            #tokenize examples
            inputs = tokenizer(examples, padding = 'max_length', truncation = True, max_length = 512, return_tensors = 'pt').to(device)

            #get the label
            labels = batch['label'].to(device)

            #pass the inputs to the model
            outputs = model(**inputs)

            #get the logits, loss
            logits = outputs['logits'].to(device)
            loss = criterion(logits, labels)

            #get the predicted label
            predictions = softmax(logits)

            #get the last hidden state
            last_hidden_state = outputs['hidden_states'][-1]
            last_hidden_state = last_hidden_state[:,0,:].to(device)

            #add the results to the tensors
            embeddings = torch.cat((embeddings, last_hidden_state), 0).to(device)
            losses = torch.cat((losses, loss), -1).to(device)
            targets = torch.cat((targets, labels), -1).to(device)
            outputs_tensor = torch.cat((outputs_tensor, torch.argmax(predictions, dim = -1)), -1).to(device)
            labels_tensor = torch.cat((labels_tensor, labels), -1).to(device)
            text.extend(examples)

    #dimensionality reduction using TSNE (will be used for clustering)
    X_embedded = TSNE(n_components = 2).fit_transform(embeddings.cpu())

    #move all tensors to CPU so they can be saved
    embeddings.cpu()
    losses.cpu()
    targets.cpu()
    outputs_tensor.cpu()
    labels_tensor.cpu()
  
    #convert tensors to pandas Series and add them all to results dataframe
    inputs = pd.Series(data = text, name = 'content')
    labels = pd.Series(data = labels_tensor.cpu(), name = 'label')
    outputs = pd.Series(data = outputs_tensor.cpu(), name = 'pred')
    losses = pd.Series(data = losses.cpu(), name = 'loss', dtype = 'float64')
    embeddings = embeddings.tolist()

    results = pd.DataFrame({
        inputs.name: inputs,
        labels.name: labels,
        outputs.name: outputs,
        dataset.string_label.name: dataset.string_label,
        dataset.string_pred.name: dataset.string_pred,
        losses.name: losses,
        'x': X_embedded[:,0],
        'y': X_embedded[:,1], 
        'embedding': embeddings
    })

    return results

def sentence_bert(embedding_model, dataset):

    #load the S-BERT model
    model = SentenceTransformer(embedding_model)

    #get the embeddings for the text
    embeddings = model.encode(dataset.content)

    #dimensionality reduction to 2D (necessary to plot the clustering later)
    X_embedded = TSNE(n_components=2).fit_transform(embeddings)
    embeddings = embeddings.tolist()

    #save results to dataframe
    results = pd.DataFrame({
        dataset.content.name: dataset.content,
        dataset.label.name: dataset.label,
        dataset.pred.name: dataset.pred,
        dataset.string_label.name: dataset.string_label,
        dataset.string_pred.name: dataset.string_pred,
        'x': X_embedded[:,0],
        'y': X_embedded[:,1],
        'embedding': embeddings,
    })

    return results

def linguistic_features(llm, embedding_model, dataset, language):

    if language == 'ar' or language == 'arabic':
        prompts, features = arabic_features(dataset.content)
    elif language == 'en' or language == 'english':
        prompts, features = english_features(llm, 
                                    dataset.content, 
                                    dataset.string_pred, 
                                    dataset.string_label)
        
    #load the S-BERT model
    model = SentenceTransformer(embedding_model)

    #get the embeddings for the text
    embeddings = model.encode(dataset.features)

    #dimensionality reduction to 2D (necessary to plot the clustering later)
    X_embedded = TSNE(n_components=2).fit_transform(embeddings)
    embeddings = embeddings.tolist()

    results = pd.DataFrame({
        'prompts': prompts,
        'features': features,
        dataset.content.name: dataset.content,
        dataset.label.name: dataset.label,
        dataset.pred.name: dataset.pred,
        dataset.string_label.name: dataset.string_label,
        dataset.string_pred.name: dataset.string_pred,
        'x': X_embedded[:,0],
        'y': X_embedded[:,1],
        'embedding': embeddings,
    })

    return results

def concatenate_embeddings(llm, embedding_model, dataset, language):
    
    #get sbert embeddings for content
    #load the S-BERT model
    model = SentenceTransformer(embedding_model)

    #get the embeddings for the text
    content_embeddings = model.encode(dataset.content)
    content_embeddings = content_embeddings.tolist()

    #get linguistic features
    if language == 'ar' or language == 'arabic':
        prompts, features = arabic_features(dataset.content)
    elif language == 'en' or language == 'english':
        prompts, features = english_features(llm, 
                                    dataset.content, 
                                    dataset.string_pred, 
                                    dataset.string_label)
        
    #get sbert embeddings of linguistic features
    feature_embeddings = model.encode(features)
    feature_embeddings = feature_embeddings.tolist()

    #concatenate lf and sb emebddings
    concatenated_embeddings = [example + feature for example, feature in zip(content_embeddings, feature_embeddings)]
    concatenated_embeddings = np.array(concatenated_embeddings)

    #dimensionality reduction to 2D (necessary to plot the clustering later)
    X_embedded = TSNE(n_components=2).fit_transform(concatenated_embeddings)
    concatenated_embeddings = concatenated_embeddings.tolist()

    results = pd.DataFrame({
        'prompts': prompts,
        'features': features,
        dataset.content.name: dataset.content,
        dataset.label.name: dataset.label,
        dataset.pred.name: dataset.pred,
        dataset.string_label.name: dataset.string_label,
        dataset.string_pred.name: dataset.string_pred,
        'x': X_embedded[:,0],
        'y': X_embedded[:,1],
        'embedding': concatenated_embeddings,
    })

    return results

def arabic_features(content):

    #initialize a cohere client
    co = cohere.Client('YOUR API KEY GOES HERE')

    features = []
    prompts = []

    prompt = '  نموذج اللغة. مهمتك هي استخراج السمات اللغوية من المثال. المثال مكتوب بالغة العامية'

    #iterate through the examples in the dataset, passing the content and the prompt to the LLM
    for example in tqdm(content):
        current = prompt + example
        response = co.chat(
            current, 
            model="command-r", 
            temperature=0.8,
            k = 50,
            max_tokens = 500,
            p = 0.90
        )
        answer = response.text
        features.append(answer)
        prompts.append(current)
    
    return prompts, features

def english_features(llm, content, string_pred, string_label):

    features = []
    prompts = []

    #use a 4-bit quantized version of the model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    #load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm)
    model = AutoModelForCausalLM.from_pretrained(
        llm,
        load_in_4bit = True,
        quantization_config = bnb_config,
        torch_dtype = torch.bfloat16,
        device_map = "cuda",
        trust_remote_code = True,
    )

    #prepare pipeline object to use for text generation later
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer = tokenizer, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda"
    )

    for example, pred, label in zip(content, string_pred, string_label):

        prompt = f' you are provided with an example from an offensive language dataset. this example was misclassified by a language model. the correct label is {label} but the model classified it as {pred}. your task is to do a linguistic and stylistic analysis to extract features from the example that may have led to the misclassification. give your output like this: <feature>: <explanation>. \n{example} \nyour output should only include the features and their explanation, nothing else'

        prompts.append()

        sequences = pipe(
            prompt,
            do_sample=True, #enables decoding strategies like top_k and top_p
            max_new_tokens=500, 
            temperature=0.9,
            top_k=50, #number of predictions to return
            top_p=0.95, #smallest probability of a generated token
            num_return_sequences=1,
        )
        current_features = sequences[0]['generated_text']
        current_features = current_features.split('[/INST]', 1)[1]
        features.append(current_features)

    return prompts, features
