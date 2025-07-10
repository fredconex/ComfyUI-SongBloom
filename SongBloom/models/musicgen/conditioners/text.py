from .base import *

import warnings
import random
import hashlib
from transformers import RobertaTokenizer, T5EncoderModel, T5Tokenizer, AutoTokenizer, XLMRobertaModel, XLMRobertaTokenizer  # type: ignore

def hash_trick(word: str, vocab_size: int) -> int:
    """Hash trick to pair each word with an index

    Args:
        word (str): word we wish to convert to an index
        vocab_size (int): size of the vocabulary
    Returns:
        int: index of the word in the embedding LUT
    """

    hash = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
    return hash % vocab_size



class PhonemeTokenizerConditioner(TextConditioner):
    def __init__(self, 
                 output_dim: int, 
                 vocab_list,
                 max_len = 600, 
                 max_sentence_per_structure = 50,
                 structure_tokens=None,
                 structure_split_tokens=[','],
                 sentence_split_tokens=['.'],
                 mode='sum',
                 structure_output_dim = 64,
                 sentence_output_dim = 64,
                 max_duration = 120,
                 interpolate = False,
                 ): 
        
        self.vocab_list = vocab_list
        self.max_len = max_len
        self.mode = mode
        self.max_sentence_per_structure = max_sentence_per_structure
        voc_size = len(self.vocab_list)
        self.interpolate = interpolate
        
        if structure_tokens is None:
            structure_tokens = [i for i in vocab_list if len(i) > 1 and i[0] == '[' and i[-1] == ']']
        self.structure_token_ids = [vocab_list.index(i) for i in structure_tokens if i in vocab_list]
        self.structure_split_token_ids = [vocab_list.index(i) for i in structure_split_tokens]
        self.sentence_split_token_ids = [vocab_list.index(i) for i in sentence_split_tokens]

        # here initialize a output_proj (nn.Embedding) layer
        # By default the first vocab is "" (null)
        if mode == 'sum':
            content_output_dim = output_dim
            sentence_output_dim = output_dim
            structure_output_dim = output_dim
        else:   # concat
            content_output_dim = output_dim - sentence_output_dim - structure_output_dim   # by default
            
        super().__init__(voc_size, content_output_dim, input_token=True, padding_idx=0)
        if self.mode != 'sum':
            self.special_emb = nn.Embedding(len(self.structure_token_ids)+len(self.structure_split_token_ids)+len(self.sentence_split_token_ids)+1, 
                                             structure_output_dim, padding_idx=0)
            
        self.blank_emb = nn.Parameter(torch.zeros(1, output_dim), requires_grad=False)

        # the first index is "empty structure" token
        self.sentence_idx_in_structure_emb = nn.Embedding(max_sentence_per_structure, sentence_output_dim, padding_idx=0) 

        # print("max_len", self.max_len)
        print(self.structure_token_ids)
        
        self.resolution = max_duration / max_len    # e.g., 120 / 600 = 0.2s 
        print(self.__class__, f"resolution = {self.resolution}")
    
    def tokenize(self, x: tp.List[tp.Optional[str]]) -> tp.Dict[str, torch.Tensor]:
        inputs = []
        for xx in x:
            xx = '' if xx is None else xx
            vocab_id = [self.vocab_list.index(item) for item in xx.split(" ") if item in self.vocab_list]
            inputs.append(torch.tensor(vocab_id).long()) # [T]        
        return inputs
    
    
    def interpolate_with_structure_duration(self, special_tokens, embeds, structure_dur):
        # embeds: [T, N]
        def sec2idx(sec):   # convert duration sec to token index
            return int(sec / self.resolution)
        
        def target_token_types2list(tokens, target_token_types):

            is_target_list = torch.any(torch.stack([tokens == i for i in target_token_types], dim=-1), dim=-1)
            is_target_list = torch.where(is_target_list)[0].tolist()
            return is_target_list
        
        structure_ids = []
        for (structure, st, et) in structure_dur:
            structure_ids.append([structure, sec2idx(st), sec2idx(et)])
            
        """
        interpolate embeddings of each structure according to its duration 
        """
        is_structure_list = target_token_types2list(special_tokens, self.structure_token_ids)
        is_structure_list.append(special_tokens.shape[-1])
        
        split_tokens = deepcopy(self.structure_split_token_ids)
        split_tokens.extend(self.sentence_split_token_ids)
        # is_split_list = target_token_types2list(special_tokens, split_tokens)
                
        
        interpolated_embeds = embeds[:is_structure_list[0]]
        for i, st in enumerate(is_structure_list[:-1]):
            # (lorry) Explain "-tmp": 
            # All structures are connected with " , " token,
            # " ," is also the final token of each structure except the final one,
            # but here we dont want to interpolate " , " token
            tmp = 1
            if i == len(is_structure_list[:-1]) - 1:  # the final structure, no need for "-1"
                tmp = 0
            
       #     print(st, is_structure_list[i+1]-tmp)
            to_interpolate = embeds[st: is_structure_list[i+1] - tmp]
            interpolate_size = structure_ids[i][2] - structure_ids[i][1] - tmp
       #     print(interpolate_size)
            
            #import pdb; pdb.set_trace()
            # print(interpolated_embeds.shape, to_interpolate.shape, interpolate_size, )
            if to_interpolate.shape[0] == 0:
                pass
            this_interpolated_embeds = F.interpolate(to_interpolate.unsqueeze(0).transpose(2, 1), 
                                        size=interpolate_size, 
                                        mode='nearest-exact').squeeze(0).transpose(1, 0)
            
            if tmp == 1:
                interpolated_embeds = torch.cat((interpolated_embeds, this_interpolated_embeds, 
                                                embeds[is_structure_list[i+1]].unsqueeze(0)), 0)
            else:
                interpolated_embeds = torch.cat((interpolated_embeds, this_interpolated_embeds), 0)
        return interpolated_embeds
            
            
    def forward(self, batch_tokens: tp.List, structure_dur = None) -> ConditionType:
        """
        Encode token_id into three types of embeddings:
        1) content embedding: phoneme only (or meaningful contents to be sung out) 
        2) structure embedding: structure / separation embeddings, including structures (verse/chorus/...), separators (. / ,)
        The two above share the same embedding layer, can be changed to separate embedding layers.
        3) sentence_idx embedding (per structure): 
        """
        embeds_batch = []
        # print(batch_tokens)
        for b in range(len(batch_tokens)):
            tokens = batch_tokens[b]  

            content_tokens = torch.zeros_like(tokens)
            special_tokens = torch.zeros_like(tokens)
            sentence_idx_in_structure_tokens = torch.zeros_like(tokens) 

            current_structure_idx = 1
            current_sentence_in_structure_idx = 1
            current_structure = 0

            for i in range(tokens.shape[-1]):
                token = tokens[i]
                if token in self.structure_token_ids:       # structure token
                    # only update structure token, leave content and sentence index token null (default 0)
                    if self.mode == 'sum':
                        special_tokens[i] = token
                    else:
                        special_tokens[i] = self.structure_token_ids.index(token) + 1
                    current_structure = token
                    current_structure_idx += 1
                    current_sentence_in_structure_idx = 1

                elif token in self.sentence_split_token_ids:    # utterance split token
                    # only update structure token, leave content and sentence index token null (default 0)
                    # add up sentence index
                    if self.mode == 'sum':
                        special_tokens[i] = token
                    else:
                        special_tokens[i] = self.sentence_split_token_ids.index(token) + 1 + len(self.structure_token_ids)
                    current_sentence_in_structure_idx += 1

                elif token in self.structure_split_token_ids:    # structure split token
                    # update structure token (current structure), content token (current token), 
                    # blank index token 
                    if self.mode == 'sum':
                        special_tokens[i] = token
                    else:
                        special_tokens[i] = self.structure_split_token_ids.index(token) + 1 + len(self.structure_token_ids) + len(self.sentence_split_token_ids)

                else:       # content tokens
                    content_tokens[i] = token
                    special_tokens[i] = current_structure
                    sentence_idx_in_structure_tokens[i] = min(current_sentence_in_structure_idx, self.max_sentence_per_structure - 1)

            # print("tokens", tokens.max(), tokens.min())
            # print("special tokens", special_tokens.max(), special_tokens.min())
            # print("sentence idx in structure", sentence_idx_in_structure_tokens.max(), sentence_idx_in_structure_tokens.min())
            device = self.output_proj.weight.device
            
            # import pdb; pdb.set_trace()
            content_embeds = self.output_proj(tokens.to(device))    # [T, N]
            if self.mode == 'sum':
                structure_embeds = self.output_proj(special_tokens.to(device))
            else:
                structure_embeds = self.special_emb(special_tokens.to(device))
            sentence_idx_embeds = self.sentence_idx_in_structure_emb(sentence_idx_in_structure_tokens.to(device))

            if self.mode == 'sum':
                embeds = content_embeds + structure_embeds + sentence_idx_embeds
            else:
                embeds = torch.cat((content_embeds, structure_embeds, sentence_idx_embeds), -1) # [T, N]
                
            if self.interpolate:
                embeds = self.interpolate_with_structure_duration(tokens, embeds, structure_dur[b])
            embeds_batch.append(embeds)

        # set batch_size = 1, [B, T, N]
        if self.max_len is not None:
            max_len = self.max_len
        else:
            max_len = max([e.shape[0] for e in embeds_batch])
        embeds, mask = self.pad_2d_tensor(embeds_batch, max_len)
        
        return embeds, mask
    
    
    def pad_2d_tensor(self, xs, max_len):
        new_tensor = []
        new_mask = []
        for x in xs:
            seq_len, dim = x.size()
            pad_len = max_len - seq_len

            if pad_len > 0:
                pad_tensor = self.blank_emb.repeat(pad_len, 1).to(x.device)  # T, D
                padded_tensor = torch.cat([x, pad_tensor], dim=0)
                mask = torch.cat((torch.ones_like(x[:, 0]), 
                                  torch.zeros_like(pad_tensor[:, 0])), 0)   # T
            elif pad_len < 0:
                padded_tensor = x[:max_len]
                mask = torch.ones_like(padded_tensor[:, 0])
            else:
                padded_tensor = x
                mask = torch.ones_like(x[:, 0])

            new_tensor.append(padded_tensor)
            new_mask.append(mask)
        # [B, T, D] & [B, T]
        return torch.stack(new_tensor, 0), torch.stack(new_mask, 0)   
