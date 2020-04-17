import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64" # Make sure this is the sama JAVA_HOME as the installed version on the previous notebook!
data_home = "/ssd2/arthur/MsMarcoTREC/"
def path(x):
    return os.path.join(data_home, x)

import pyserini
import tqdm
import jnius_config
jnius_config.add_options('-Xmx16G') # Adjust to your machine. Probably less than 16G.
from pyserini.search import pysearch
import subprocess
from tqdm.auto import tqdm
import random
import pickle
import sys
import unicodedata
import string
import re
import os
from collections import defaultdict
import math


index_path = path("lucene-index.msmarco-doc.pos+docvectors+rawdocs")
searcher = pysearch.SimpleSearcher(index_path)
relevant_docs = defaultdict(lambda:[])
for file in [path("qrels/msmarco-doctrain-qrels.tsv"), path("qrels/msmarco-docdev-qrels.tsv")]:
    for line in open(file):
        query_id, _, doc_id, rel = line.split()
        assert rel == "1"
        relevant_docs[query_id].append(doc_id)                            

pattern = re.compile('([^\s\w]|_)+')

anserini_top_10 = defaultdict(lambda:[])
searcher.set_bm25_similarity(0.9, 0.4)
pairs_per_split = defaultdict(lambda: [])
threads = 42 # Number of Threads to use when retrieving
k = 10       # Number of documents to retrieve 
neg_samples = 2 # Number of negatives samples to use
batch_size = 10000 # Batch size for each retrieval step on Anserini

query_texts = dict()
for split in ["train", "dev"]:
    file_path = path(f"queries/msmarco-doc{split}-queries.tsv")
    run_search=True
    if os.path.isfile(file_path):
        print(f"Already found file {file_path}. Cowardly refusing to run this again. Will only load querytexts.")
        pairs_per_split[split] = pickle.load(open(path(f"{split}_triples.pkl"), 'rb'))
        run_search = False
    number_of_queries = int(subprocess.run(f"wc -l {file_path}".split(), capture_output=True).stdout.split()[0])
    number_of_batches = math.ceil(number_of_queries/batch_size)
    pbar = tqdm(total=number_of_batches, desc="Retrieval batches")
    queries = []
    query_ids = []
    for idx, line in enumerate(open(file_path, encoding="utf-8")):
        query_id, query = line.strip().split("\t")
        query_ids.append(query_id)
        query = unicodedata.normalize("NFKD", query) # Force queries into UTF-8
        query = pattern.sub(' ',query) # Remove non-ascii characters. It clears up most of the issues we may find on the query datasets
        query_texts[query_id] = query
        if run_search is False:
            continue
        queries.append(query)
        if len(queries) == batch_size or idx == number_of_queries-1:
            results = searcher.batch_search(queries, query_ids, k=k, threads=threads)
            pbar.update()
            for query, query_id in zip(queries, query_ids):
                retrieved_docs_ids = [hit.docid for hit in results[query_id]]
                relevant_docs_for_query = relevant_docs[query_id]
                retrieved_non_relevant_documents = set(retrieved_docs_ids).difference(set(relevant_docs_for_query))
                  
                if len(retrieved_non_relevant_documents) < 2:
                    print(f"query {query} has less than 2 retrieved docs.")
                    continue
                random_negative_samples = random.sample(retrieved_non_relevant_documents, neg_samples)
                pairs_per_split[split] += [(query_id, doc_id, 1) for doc_id in relevant_docs_for_query]
                pairs_per_split[split] += [(query_id, doc_id, 0) for doc_id in random_negative_samples]
            queries = []
            query_ids = []
    pickle.dump(pairs_per_split[split], open(path(f"{split}_triples.pkl"), 'wb'))
    pbar.close()


from torch.utils.data import Dataset
import torch

# This is our main Dataset class.
class MsMarcoDataset(Dataset):
    def __init__(self,
                 samples,
                 tokenizer,
                 searcher,
                 split,
                 tokenizer_batch=8000):
        '''Initialize a Dataset object. 
        Arguments:
            samples: A list of samples. Each sample should be a tuple with (query_id, doc_id, <label>), where label is optional
            tokenizer: A tokenizer object from Hugging Face's Tokenizer lib. (need to implement encode_batch())
            searcher: A PySerini Simple Searcher object. Should implement the .doc() method
            split: A strong indicating if we are in a train, dev or test dataset.
            tokenizer_batch: How many samples to be tokenized at once by the tokenizer object.
            The biggest bottleneck is the searcher, not the tokenizer.
        '''
        self.searcher = searcher
        self.split = split
        # If we already have the data pre-computed, we shouldn't need to re-compute it.
        self.split = split
        if (os.path.isfile(path(f"{split}_msmarco_samples.tsv"))
                and os.path.isfile(path(f"{split}_msmarco_offset.pkl"))
                and os.path.isfile(path(f"{split}_msmarco_index.pkl"))):
            print("Already found every meaningful file. Cowardly refusing to re-compute.")
            self.samples_offset_dict = pickle.load(open(path(f"{split}_msmarco_offset.pkl"), 'rb'))
            self.index_dict = pickle.load(open(path(f"{split}_msmarco_index.pkl"), 'rb'))
            return
        self.tokenizer = tokenizer
        print("Loading and tokenizing dataset...")
        self.samples_offset_dict = dict()
        self.index_dict = dict()

        self.samples_file = open(path(f"{split}_msmarco_samples.tsv"),'w',encoding="utf-8")
        self.processed_samples = 0
        query_batch = []
        doc_batch = []
        sample_ids_batch = []
        labels_batch = []
        number_of_batches = math.ceil(len(samples) // tokenizer_batch)
        # A progress bar to display how far we are.
        batch_pbar = tqdm(total=number_of_batches, desc="Tokenizer batches")
        for i, sample in enumerate(samples):
            if split=="train" or split == "dev":
                label = sample[2]
                labels_batch.append(label)
            query_batch.append(query_texts[sample[0]])
            doc_batch.append(self._get_document_content_from_id(sample[1]))
            sample_ids_batch.append(f"{sample[0]}_{sample[1]}")
            #If we hit the number of samples for this batch OR this is the last sample
            if len(query_batch) == tokenizer_batch or i == len(samples) - 1:
                self._tokenize_and_dump_batch(doc_batch, query_batch, labels_batch, sample_ids_batch)
                batch_pbar.update()
                query_batch = []
                doc_batch = []
                sample_ids_batch = []
                if split == "train" or split == "dev":
                    labels_batch = []
        batch_pbar.close()
        # Dump files in disk, so we don't need to go over it again.
        self.samples_file.close()
        pickle.dump(self.index_dict, open(path(f"{self.split}_msmarco_index.pkl"), 'wb'))
        pickle.dump(self.samples_offset_dict, open(path(f"{self.split}_msmarco_offset.pkl"), 'wb'))

    def _tokenize_and_dump_batch(self, doc_batch, query_batch, labels_batch,
                                 sample_ids_batch):
        '''tokenizes and dumps the samples in the current batch
        It also store the positions from the current file into the samples_offset_dict.
        '''
        # Use the tokenizer object
        tokens = self.tokenizer.encode_batch(list(zip(query_batch, doc_batch)))
        for idx, (sample_id, token) in enumerate(zip(sample_ids_batch, tokens)):
            #BERT supports up to 512 tokens. If we have more than that, we need to remove some tokens from the document
            if len(token.ids) >= 512:
                token_ids = token.ids[:511]
                token_ids.append(tokenizer.token_to_id("[SEP]"))
                segment_ids = token.type_ids[:512]
            # With less tokens, we need to "pad" the vectors up to 512.
            else:
                padding = [0] * (512 - len(token.ids))
                token_ids = token.ids + padding
                segment_ids = token.type_ids + padding
            # How far in the file are we? This is where we need to go to find the documents later.
            file_location = self.samples_file.tell()
            # If we have labels
            if self.split=="train" or split == "dev":
                self.samples_file.write(f"{sample_id}\t{token_ids}\t{segment_ids}\t{labels_batch[idx]}\n")
            else:
                self.samples_file.write(f"{sample_id}\t{token_ids}\t{segment_ids}\n")
            self.samples_offset_dict[sample_id] = file_location
            self.index_dict[self.processed_samples] = sample_id
            self.processed_samples += 1

    def _get_document_content_from_id(self, doc_id):
        '''Get the raw text value from the doc_id
        There is probably an easier way to do that, but this works.
        '''
        doc_text = self.searcher.doc(doc_id).lucene_document().getField("raw").stringValue()
        return doc_text[7:-8]

    def __getitem__(self, idx):
        '''Returns a sample with index idx
        DistilBERT does not take into account segment_ids. (indicator if the token comes from the query or the document) 
        However, for the sake of completness, we are including it here, together with the attention mask
        position_ids, with the positional encoder, is not needed. It's created for you inside the model.
        '''
        if isinstance(idx, int):
            idx = self.index_dict[idx]
        with open(path(f"{self.split}_msmarco_samples.tsv"), 'r', encoding="utf-8") as inf:
            inf.seek(self.samples_offset_dict[idx])
            line = inf.readline().split("\t")
            try:
                sample_id = line[0]
                input_ids = eval(line[1])
                token_type_ids = eval(line[2])
                input_mask = [1] * 512
            except:
                print(line, idx)
                raise IndexError
            # If it's a training dataset, we also have a label tag.
            if split=="train" or split == "dev":
                label = int(line[3])
                return (torch.tensor(input_ids, dtype=torch.long),
                        torch.tensor(input_mask, dtype=torch.long),
                        torch.tensor(token_type_ids, dtype=torch.long),
                        torch.tensor([label], dtype=torch.long))
            return (torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(input_mask, dtype=torch.long),
                    torch.tensor(token_type_ids, dtype=torch.long))
    def __len__(self):
        return len(self.samples_offset_dict)

from transformers import DistilBertForSequenceClassification
from torch.utils.data import DataLoader
from tokenizers import BertWordPieceTokenizer

tokenizer = BertWordPieceTokenizer("/ssd2/arthur/bert-axioms/tokenizer/bert-base-uncased-vocab.txt", lowercase=True)

train_dataset = MsMarcoDataset(pairs_per_split["train"], tokenizer, searcher, split = "train")
dev_dataset = MsMarcoDataset(pairs_per_split["dev"], tokenizer, searcher, split = "dev")

from transformers import AdamW, get_linear_schedule_with_warmup

# With these configurations, on DeepIR, it takes ~3h/batch to train, with ~2batches/s
GPUS_TO_USE = [2,4,5,6,7] # If you have multiple GPUs, pick the ones you want to use.
number_of_cpus = 24 # Number of CPUS to use when loading your dataset.
n_epochs = 2 # How may passes over the whole dataset to complete
weight_decay = 0.0 # Some papers define a weight decay, meaning, the weights on some layers will decay slower overtime. By default, we don't do this.
lr = 0.00005 # Learning rate for the fine-tunning.
warmup_proportion = 0.1 # Percentage of training steps to perform before we start to decrease the learning rate.
steps_to_print = 500 # How many steps to wait before printing loss
steps_to_eval = 1000 # How many steps to wait before running an eval step

# This is our base model
try:
    del model
    torch.cuda.empty_cache() # Make sure we have a clean slate
except:
    pass
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

if torch.cuda.is_available():
    # Asssign the model to GPUs, specifying to use Data parallelism.
    model = torch.nn.DataParallel(model, device_ids=GPUS_TO_USE)
    # The main model should be on the first GPU
    device = torch.device(f"cuda:{GPUS_TO_USE[0]}") 
    model.to(device)
    # For a 1080Ti, 16 samples fit on a GPU confortably. So, the train batch size will be 16*the number of GPUS
    train_batch_size = len(GPUS_TO_USE) * 16
    print(f"running on {len(GPUS_TO_USE)} GPUS, on {train_batch_size}-sized batches")
else:
    print("Are you sure about it? We will try to run this in CPU, but it's a BAD idea...")
    device = torch.device("cpu")
    train_batch_size = 16
    model.to(device)

# A data loader is a nice device for generating batches for you easily.
# It receives any object that implementes __getitem__(self, idx) and __len__(self)
train_data_loader = DataLoader(train_dataset, batch_size=train_batch_size, num_workers=number_of_cpus,shuffle=True)
dev_data_loader = DataLoader(dev_dataset, batch_size=32, num_workers=number_of_cpus,shuffle=True)

#how many optimization steps to run, given the NUMBER OF BATCHES. (The len of the dataloader is the number of batches).
num_train_optimization_steps = len(train_data_loader) * n_epochs

#which layers will not have a linear weigth decay when training
no_decay = ['bias', 'LayerNorm.weight']

#all parameters to be optimized by our fine tunning.
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any( nd in n for nd in no_decay)], 'weight_decay': weight_decay},
    {'params': [p for n, p in model.named_parameters() if any( nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

#We use the AdamW optmizer here.
optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8) 

# How many steps to wait before we start to decrease the learning rate
warmup_steps = num_train_optimization_steps * warmup_proportion 
# A scheduler to take care of the above.
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps)
print(f"*********Total optmization steps: {num_train_optimization_steps}*********")


import warnings
import numpy as np
import datetime
from sklearn.metrics import f1_score, average_precision_score, accuracy_score, roc_auc_score


global_step = 0 # Number of steps performed so far
tr_loss = 0.0 # Training loss
model.zero_grad() # Initialize gradients to 0

for _ in tqdm(range(n_epochs), desc="Epochs"):
    for step, batch in tqdm(enumerate(train_data_loader), desc="Batches", total=len(train_data_loader)):
        model.train()
        # get the batch inpute
        inputs = {
            'input_ids': batch[0].to(device),
            'attention_mask': batch[1].to(device),
            'labels': batch[3].to(device)
        }
        # Run through the network.
        
        with warnings.catch_warnings():
            # There is a very annoying warning here when we are using multiple GPUS,
            # As described here: https://github.com/huggingface/transformers/issues/852.
            # We can safely ignore this.
            warnings.simplefilter("ignore")
            outputs = model(**inputs)
        loss = outputs[0]

        loss = loss.sum()/len(model.device_ids) # Average over all GPUS.
        # Clipping gradients. Avoud gradient explosion, if the gradient is too large.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # Backward pass on the network
        loss.backward()
        tr_loss += loss.item()
        # Run the optimizer with the gradients
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        if step % steps_to_print == 0:
            # Logits is the actual output from the network. 
            # This is the probability of being relevant or not.
            # You can check its shape (Should be a vector sized 2) with logits.shape()
            logits = outputs[1]
            # Send the logits to the CPU and in numpy form. Easier to check what is going on.
            preds = logits.detach().cpu().numpy()
            
            # Bring the labels to CPU too.
            out_label_ids = inputs['labels'].detach().cpu().numpy().flatten()
            tqdm.write(f"Train ROC: {roc_auc_score(out_label_ids, preds[:, 1])}")
            
            #Get the actual relevance label, not only probability.
            preds = np.argmax(preds, axis=1)
            tqdm.write(f"Train accuracy: {accuracy_score(out_label_ids, preds)}")
            tqdm.write(f"Training loss: {loss.item()}")
            tqdm.write(f"Learning rate: {scheduler.get_last_lr()[0]}")
        global_step += 1
        
        # Run an evluation step over the eval dataset. Let's see how we are going.
        if global_step%steps_to_eval == 0:
            eval_loss = 0.0
            nb_eval_steps = 0
            preds = None
            out_label_ids = None
            for batch in dev_data_loader, desc="Dev batch":
                model.eval()
                with torch.no_grad(): # Avoid upgrading gradients here
                    inputs = {'input_ids': batch[0].to(device),
                      'attention_mask': batch[1].to(device),
                      'labels': batch[3].to(device)}
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        outputs = model(**inputs)
                    tmp_eval_loss, logits = outputs[:2] # Logits is the actual output. Probabilities between 0 and 1.
                    eval_loss += tmp_eval_loss.mean().item()
                    nb_eval_steps += 1
                    # Concatenate all outputs to evaluate in the end.
                    if preds is None:
                        preds = logits.detach().cpu().numpy() # PRedictions into numpy mode
                        out_label_ids = inputs['labels'].detach().cpu().numpy().flatten() # Labels assigned by model
                    else:
                        batch_predictions = logits.detach().cpu().numpy()
                        preds = np.append(preds, batch_predictions, axis=0)
                        out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy().flatten(), axis=0)
                eval_loss = eval_loss / nb_eval_steps
            results = {}
            results["ROC Dev"] = roc_auc_score(out_label_ids, preds[:, 1])
            preds = np.argmax(preds, axis=1)
            results["Acuracy Dev"] = accuracy_score(out_label_ids, preds)
            results["F1 Dev"] = f1_score(out_label_ids, preds)
            results["AP Dev"] = average_precision_score(out_label_ids, preds)
            tqdm.write("***** Eval results *****")
            for key in sorted(results.keys()):
                tqdm.write(f"  {key} = {str(results[key])}")
            output_dir = path(f"checkpoints/checkpoint-{global_step}")
            if not os.path.isdir(output_dir):
                os.makedirs(path(output_dir))
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(output_dir)

# Save final model 
output_dir = path(f"models/distilBERT-{str(datetime.date.today())}")
if not os.path.isdir(output_dir):
    os.makedirs(path(output_dir))
model_to_save = model.module if hasattr(model, 'module') else model
model_to_save.save_pretrained(output_dir)