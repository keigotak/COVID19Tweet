import torch
import argparse

from transformers import *

from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
from AbstractEmbedding import AbstractEmbedding


class HaggingFaceEmbeddings(AbstractEmbedding):
    def __init__(self, device, model):
        super(HaggingFaceEmbeddings, self).__init__(device=device)
        self.model_keys = self.get_model_keys()
        MODELS = {'bert-base-uncased': (BertModel, BertTokenizer, 'bert-base-uncased'),
                  'openai-gpt': (OpenAIGPTModel, OpenAIGPTTokenizer, 'openai-gpt'),
                  'transfo-xl-wt103': (TransfoXLModel, TransfoXLTokenizer, 'transfo-xl-wt103'),
                  'gpt2': (GPT2Model, GPT2Tokenizer, 'gpt2'),
                  'xlm-mlm-enfr-1024': (XLMModel, XLMTokenizer, 'xlm-mlm-enfr-1024'),
                  'xlnet-base-cased': (XLNetModel, XLNetTokenizer, 'xlnet-base-cased'),
                  'roberta-base': (RobertaModel, RobertaTokenizer, 'roberta-base'),
                  'distilbert-base-uncased': (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased'),
                  'ctrl': (CTRLModel, CTRLTokenizer, 'ctrl'),
                  'distilbert-base-cased': (DistilBertModel, DistilBertTokenizer, 'distilbert-base-cased'),
                  'camembert': (CamembertModel, CamembertTokenizer, 'camembert-base'),
                  'albert-base-v2': (AlbertModel, AlbertTokenizer, 'albert-base-v2'),
                  'xlm-roberta-base': (XLMRobertaModel, XLMRobertaTokenizer, 'xlm-roberta-base'),
                  'flaubert_base_cased': (FlaubertModel, FlaubertTokenizer, 'flaubert/flaubert_base_cased'),
                  'bart-large': (BartModel, BartTokenizer, 'facebook/bart-large'),
                  't5-small': (T5Model, T5Tokenizer, 't5-small'),
                  'electra-small-discriminator': (ElectraModel, ElectraTokenizer, 'google/electra-small-discriminator'),
                  # DiaploGPT
                  'reformer-crime-and-punishment': (ReformerModel, ReformerTokenizer, 'google/reformer-crime-and-punishment'),
                  'opus-mt-en-ROMANCE': (MarianMTModel, MarianTokenizer, 'Helsinki-NLP/opus-mt-en-ROMANCE'),
                  'longformer-base-4096': (LongformerModel, LongformerTokenizer, 'allenai/longformer-base-4096'),
                  'retribert': (RetriBertModel, RetriBertTokenizer, 'distilbert-base-uncased'),
                  'mobilebert-uncased': (MobileBertModel, MobileBertTokenizer, 'google/mobilebert-uncased')
                  }

        if model not in self.model_keys:
            assert '{} is not in keys'.format(model)


        self.model_name = MODELS[model][2]
        self.tokenizer = MODELS[model][1].from_pretrained(self.model_name)
        self.model = MODELS[model][0].from_pretrained(self.model_name)
        self.model.eval()  # disable dropout (or leave in train mode to finetune)
        self.model.to(self.device)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.embedding_dim = self.model.config.hidden_size

        parser = argparse.ArgumentParser()
        parser.add_argument('--bpe-codes',
                            default="../data/models/BERTweet_base_transformers/bpe.codes",
                            required=False,
                            type=str,
                            help='path to fastBPE BPE'
                            )
        args = parser.parse_args()
        self.bpe = fastBPE(args)
        self.max_seq_length = 256

    def forward(self, sentences):
        with_bpe = False
        if with_bpe:
            all_input_ids = []
            for sentence in sentences:
                sentence = '<s> ' + self.bpe.encode(sentence) + ' </s>'
                # Encode the line using fastBPE & Add prefix <s> and suffix </s>
                input_ids = self.tokenizer(sentence)
                all_input_ids.append(input_ids['input_ids'])
            # Padding ids
            max_seq_length = max(map(len, all_input_ids))
            pad_all_input_ids = [input_ids + [self.pad_token_id] * (max_seq_length - len(input_ids)) for input_ids
                                 in all_input_ids]
            pad_all_input_ids = torch.tensor([pad_all_input_ids], dtype=torch.long).squeeze(0)
        else:
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = '<pad>'
            pad_all_input_ids = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").data['input_ids']

        # Extract features
        with torch.no_grad():
            features = self.model(pad_all_input_ids.to(self.device))

        return features[0]

    @staticmethod
    def get_model_keys():
        return ['bert-base-uncased', 'openai-gpt', 'transfo-xl-wt103', 'gpt2', 'xlm-mlm-enfr-1024',
                           'xlnet-base-cased', 'roberta-base', 'distilbert-base-uncased', 'ctrl',
                           'distilbert-base-cased', 'camembert', 'albert-base-v2', 'xlm-roberta-base',
                           'flaubert_base_cased', 'bart-large', 'electra-small-discriminator',
                           'mobilebert-uncased']



if __name__ == '__main__':
    sentences = ['Official death toll from #covid19 in the United Kingdom is now GREATER than: Germany + Poland + Switzerland + Austria + Portugal + Greece + Sweden + Finland + Norway + Ireland... COMBINED. UK: 67.5 Million (233 dead) Above group: 185 Million (230 dead) HTTPURL',
                 'Dearest Mr. President @USER 1,169 coronavirus deaths in the US in 24 hours (?) Covid19 pandemic is an international crime from China - not a nature disasster! Please use your authorities to protect your people and world against China! #ChinaHasToCompensateAll',
                 'Latest Updates March 20 ⚠️5274 new cases and 38 new deaths in the United States Illinois: Governo Pritzker issues "stay at home" order for all residents New York: Governor Cuomo orders 100% of all non-essential workers to stay home Penns...Source ( /coronavirus/country/us/ )',
                 '真把公主不当干部 BREAKING: 21 people on Grand Princess cruise ship docked off the California coast tested positive for coronavirus, including 19 crew members and two passengers, Vice Pres. Mike Pence says. 24 people tested negative. HTTPURL HTTPURL',
                 "SC has first two presumptive cases of coronavirus , DHEC confirms HTTPURL via @USER :cry:"]
    for model in HaggingFaceEmbeddings.get_model_keys():
        emb = HaggingFaceEmbeddings(device='cuda:1', model=model)
        emb(sentences)
