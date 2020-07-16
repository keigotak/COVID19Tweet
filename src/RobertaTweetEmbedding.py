import torch
from transformers import *
from tokenizers import *


# https://github.com/VinAIResearch/BERTweet
# https://arxiv.org/abs/2005.10200


import torch

# Load BERTweet-base in fairseq
from fairseq.models.roberta import RobertaModel

BERTweet = RobertaModel.from_pretrained('/Absolute-path-to/BERTweet_base_fairseq', checkpoint_file='model.pt')
BERTweet.eval()  # disable dropout (or leave in train mode to finetune)

# Incorporate the BPE encoder into BERTweet-base
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq import options

parser = options.get_preprocessing_parser()
parser.add_argument('--bpe-codes', type=str, help='path to fastBPE BPE',
                    default="/Absolute-path-to/BERTweet_base_fairseq/bpe.codes")
args = parser.parse_args()
BERTweet.bpe = fastBPE(args)  # Incorporate the BPE encoder into BERTweet

# INPUT TEXT IS TOKENIZED!
line = "SC has first two presumptive cases of coronavirus , DHEC confirms HTTPURL via @USER :cry:"

# Extract the last layer's features
subwords = BERTweet.encode(line)
last_layer_features = BERTweet.extract_features(subwords)
assert last_layer_features.size() == torch.Size([1, 21, 768])

# Extract all layer's features (layer 0 is the embedding layer)
all_layers = BERTweet.extract_features(subwords, return_all_hiddens=True)
assert len(all_layers) == 13
assert torch.all(all_layers[-1] == last_layer_features)

# Filling marks
masked_line = 'SC has first two presumptive cases of  <mask> , DHEC confirms HTTPURL via @USER :cry:'
topk_filled_outputs = BERTweet.fill_mask(masked_line, topk=5)
for candidate in topk_filled_outputs:
    print(candidate)
    # ('SC has first two presumptive cases of coronavirus , DHEC confirms HTTPURL via @USER :cry:', 0.8643638491630554, 'coronavirus')
    # ('SC has first two presumptive cases of Coronavirus , DHEC confirms HTTPURL via @USER :cry:', 0.04520644247531891, 'Coronavirus')
    # ('SC has first two presumptive cases of #coronavirus , DHEC confirms HTTPURL via @USER :cry:', 0.035870883613824844, '#coronavirus')
    # ('SC has first two presumptive cases of #COVID19 , DHEC confirms HTTPURL via @USER :cry:', 0.029708299785852432, '#COVID19')
    # ('SC has first two presumptive cases of #Coronavirus , DHEC confirms HTTPURL via @USER :cry:', 0.005226477049291134, '#Coronavirus')


# #------------
# import torch
# # import argparse
# from pathlib import Path
#
# from transformers import RobertaConfig
# from transformers import RobertaModel
#
# from fairseq.data.encoders.fastbpe import fastBPE
# from fairseq.data import Dictionary
#
# # Load model
# path_model = Path("../data/models/BERTweet_base_transformers")
# config = RobertaConfig.from_pretrained(
#     str(Path("../data/models/BERTweet_base_transformers/config.json").resolve())
# )
# BERTweet = RobertaModel.from_pretrained(
#     str(Path("../data/models/BERTweet_base_transformers/model.bin").resolve()),
#     config=config
# )
#
# # # Tokenizers provides ultra-fast implementations of most current tokenizers:
# # from tokenizers import (ByteLevelBPETokenizer,
# #                         CharBPETokenizer,
# #                         SentencePieceBPETokenizer,
# #                         BertWordPieceTokenizer)
# # tokenizer = CharBPETokenizer(vocab_file=str(Path("../data/models/BERTweet_base_transformers/dict.txt.woid.json").resolve()),
# #                              merges_file=str(Path("../data/models/BERTweet_base_transformers/bpe.codes.woid").resolve()))
#
# # Load BPE encoder
# parser = argparse.ArgumentParser()
# parser.add_argument('--bpe-codes',
#     default=str(Path("../models/BERTweet_base_transformers/bpe.codes").absolute()),
#     required=False,
#     type=str,
#     help='path to fastBPE BPE'
# )
# args = parser.parse_args()
# bpe = fastBPE(args)
#
# # Load the dictionary
# vocab = Dictionary()
# vocab.add_from_file(str(Path("../models/BERTweet_base_transformers/dict.txt").absolute()))
#
# # INPUT TEXT IS TOKENIZED!
# line = "SC has first two presumptive cases of coronavirus , DHEC confirms HTTPURL via @USER :cry:"
#
# # Encode the line using fastBPE & Add prefix <s> and suffix </s>
# subwords = '<s> ' + tokenizer.encode(line) + ' </s>'
#
# # Map subword tokens to corresponding indices in the dictionary
# input_ids = vocab.encode_line(subwords, append_eos=False, add_if_not_exist=False).long().tolist()
#
# # Convert into torch tensor
# all_input_ids = torch.tensor([input_ids], dtype=torch.long)
#
# # Extract features
# with torch.no_grad():
#     features = BERTweet(all_input_ids)
#
# # Represent each word by the contextualized embedding of its first subword token
# # i. Get indices of the first subword tokens of words in the input sentence
# listSWs = subwords.split()
# firstSWindices = []
# for ind in range(1, len(listSWs) - 1):
#     if not listSWs[ind - 1].endswith("@@"):
#         firstSWindices.append(ind)
#
# # ii. Extract the corresponding contextualized embeddings
# words = line.split()
# assert len(firstSWindices) == len(words)
# vectorSize = features[0][0, 0, :].size()[0]
# for word, index in zip(words, firstSWindices):
#     print(word + " --> " + " ".join([str(features[0][0, index, :][_ind].item()) for _ind in range(vectorSize)]))
#     # print(word + " --> " + listSWs[index] + " --> " + " ".join([str(features[0][0, index, :][_ind].item()) for _ind in range(vectorSize)]))




# # Transformers has a unified API
# # for 10 transformer architectures and 30 pretrained weights.
# #          Model          | Tokenizer          | Pretrained weights shortcut
# MODELS = [(BertModel,       BertTokenizer,       'bert-base-uncased'),
#           (OpenAIGPTModel,  OpenAIGPTTokenizer,  'openai-gpt'),
#           (GPT2Model,       GPT2Tokenizer,       'gpt2'),
#           (CTRLModel,       CTRLTokenizer,       'ctrl'),
#           (TransfoXLModel,  TransfoXLTokenizer,  'transfo-xl-wt103'),
#           (XLNetModel,      XLNetTokenizer,      'xlnet-base-cased'),
#           (XLMModel,        XLMTokenizer,        'xlm-mlm-enfr-1024'),
#           (DistilBertModel, DistilBertTokenizer, 'distilbert-base-cased'),
#           (RobertaModel,    RobertaTokenizer,    'roberta-base'),
#           (XLMRobertaModel, XLMRobertaTokenizer, 'xlm-roberta-base'),
#          ]
#
# # Tokenizers provides ultra-fast implementations of most current tokenizers:
# >>> from tokenizers import (ByteLevelBPETokenizer,
#                             CharBPETokenizer,
#                             SentencePieceBPETokenizer,
#                             BertWordPieceTokenizer)
# # Ultra-fast => they can encode 1GB of text in ~20sec on a standard server's CPU
# # Tokenizers can be easily instantiated from standard files
# >>> tokenizer = BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=True)
#
# # To use TensorFlow 2.0 versions of the models, simply prefix the class names with 'TF', e.g. `TFRobertaModel` is the TF 2.0 counterpart of the PyTorch model `RobertaModel`
#
# # Let's encode some text in a sequence of hidden-states using each model:
# for model_class, tokenizer_class, pretrained_weights in MODELS:
#     # Load pretrained model/tokenizer
#     tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
#     model = model_class.from_pretrained(pretrained_weights)
#
#     # Encode text
#     input_ids = torch.tensor([tokenizer.encode("Here is some text to encode", add_special_tokens=True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
#     with torch.no_grad():
#         last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples
#
# # Each architecture is provided with several class for fine-tuning on down-stream tasks, e.g.
# BERT_MODEL_CLASSES = [BertModel, BertForPreTraining, BertForMaskedLM, BertForNextSentencePrediction,
#                       BertForSequenceClassification, BertForTokenClassification, BertForQuestionAnswering]
#
# # All the classes for an architecture can be initiated from pretrained weights for this architecture
# # Note that additional weights added for fine-tuning are only initialized
# # and need to be trained on the down-stream task
# pretrained_weights = 'bert-base-uncased'
# tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
# for model_class in BERT_MODEL_CLASSES:
#     # Load pretrained model/tokenizer
#     model = model_class.from_pretrained(pretrained_weights)
#
#     # Models can return full list of hidden-states & attentions weights at each layer
#     model = model_class.from_pretrained(pretrained_weights,
#                                         output_hidden_states=True,
#                                         output_attentions=True)
#     input_ids = torch.tensor([tokenizer.encode("Let's see all hidden-states and attentions on this text")])
#     all_hidden_states, all_attentions = model(input_ids)[-2:]
#
#     # Models are compatible with Torchscript
#     model = model_class.from_pretrained(pretrained_weights, torchscript=True)
#     traced_model = torch.jit.trace(model, (input_ids,))
#
#     # Simple serialization for models and tokenizers
#     model.save_pretrained('./directory/to/save/')  # save
#     model = model_class.from_pretrained('./directory/to/save/')  # re-load
#     tokenizer.save_pretrained('./directory/to/save/')  # save
#     tokenizer = BertTokenizer.from_pretrained('./directory/to/save/')  # re-load
#
#
#
#
