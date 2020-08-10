# https://github.com/VinAIResearch/BERTweet
# https://arxiv.org/abs/2005.10200

import torch
import argparse

from transformers import RobertaConfig
from transformers import RobertaModel

from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
from AbstractEmbedding import AbstractEmbedding


class RobertaTweetEmbedding(AbstractEmbedding):
    def __init__(self, device):
        super(RobertaTweetEmbedding, self).__init__(device=device)
        self.config = RobertaConfig.from_pretrained('../data/models/BERTweet_base_transformers/config.json')
        self.model = RobertaModel.from_pretrained('../data/models/BERTweet_base_transformers/model.bin', config=self.config)
        self.model.eval()  # disable dropout (or leave in train mode to finetune)
        self.model.to(self.device)
        self.pad_token_id = self.config.pad_token_id

        # Load BPE encoder
        parser = argparse.ArgumentParser()
        parser.add_argument('--bpe-codes',
                            default="../data/models/BERTweet_base_transformers/bpe.codes",
                            required=False,
                            type=str,
                            help='path to fastBPE BPE'
                            )
        args = parser.parse_args()
        self.bpe = fastBPE(args)

        # Load the dictionary
        self.vocab = Dictionary()
        self.vocab.add_from_file("../data/models/BERTweet_base_transformers/dict.txt")

    def forward(self, sentences):
        all_input_ids = []
        for sentence in sentences:
            # Encode the line using fastBPE & Add prefix <s> and suffix </s>
            subwords = '<s> ' + self.bpe.encode(sentence) + ' </s>'

            # Map subword tokens to corresponding indices in the dictionary
            input_ids = self.vocab.encode_line(subwords, append_eos=False, add_if_not_exist=False).long().tolist()
            all_input_ids.append(input_ids)

        # Padding ids
        max_seq_length = max(map(len, all_input_ids))
        pad_all_input_ids = [input_ids + [self.pad_token_id] * (max_seq_length - len(input_ids)) for input_ids in all_input_ids]

        # Extract features
        with torch.no_grad():
            features = self.model(torch.tensor([pad_all_input_ids], dtype=torch.long).squeeze(0).to(self.device))

        return features[0]


if __name__ == '__main__':
    sentences = ['Official death toll from #covid19 in the United Kingdom is now GREATER than: Germany + Poland + Switzerland + Austria + Portugal + Greece + Sweden + Finland + Norway + Ireland... COMBINED. UK: 67.5 Million (233 dead) Above group: 185 Million (230 dead) HTTPURL',
                 'Dearest Mr. President @USER 1,169 coronavirus deaths in the US in 24 hours (?) Covid19 pandemic is an international crime from China - not a nature disasster! Please use your authorities to protect your people and world against China! #ChinaHasToCompensateAll',
                 'Latest Updates March 20 ⚠️5274 new cases and 38 new deaths in the United States Illinois: Governo Pritzker issues "stay at home" order for all residents New York: Governor Cuomo orders 100% of all non-essential workers to stay home Penns...Source ( /coronavirus/country/us/ )',
                 '真把公主不当干部 BREAKING: 21 people on Grand Princess cruise ship docked off the California coast tested positive for coronavirus, including 19 crew members and two passengers, Vice Pres. Mike Pence says. 24 people tested negative. HTTPURL HTTPURL',
                 "SC has first two presumptive cases of coronavirus , DHEC confirms HTTPURL via @USER :cry:"]
    emb = RobertaTweetEmbedding(device='cuda:1')
    emb(sentences)
