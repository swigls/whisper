from typing import Optional, List, Tuple, Iterable
import numpy as np

import whisper
from whisper.tokenizer import get_tokenizer
from whisper.normalizers import EnglishTextNormalizer
from whisper.subword_trie import SubwordTrie, SingletonTokenizer

import os
import sys
import treevizer


def set_glossary(trie: SubwordTrie,
                 line: str,
                 augmentations = [],
                 split_index : Optional[int]=None,  # only used by bpe_single_cut
                 ):
    if len(augmentations) == 0:
        # add a space at the beginning of a glossary to make it a valid word
        line = ' '+line
        if split_index:
            split_index += 1  # add 1 to account for the space at the beginning
            assert 0 < split_index < len(line), f"split_index out of range: {split_index}"
            tokens_pre = SingletonTokenizer.encode(line[:split_index])
            tokens_post = SingletonTokenizer.encode(line[split_index:])
            tokens = tokens_pre + tokens_post
        else:
            tokens = SingletonTokenizer.encode(line)
        # print(f'{line}: {[SingletonTokenizer.decode([token]) for token in tokens]}')
        trie.set(tokens)
        return
    
    set_glossary(trie, line, augmentations[1:])
    match augmentations[0]:
        case 'first_capitalize':
            if line[0].isalpha():
                if line[0].isupper():
                    line = line[0].lower() + line[1:]
                else:
                    line = line[0].upper() + line[1:]
            else:
                assert False, f"The first letter must be an alphabet: {line}"
            set_glossary(trie, line, augmentations[1:])
        case 'bpe_single_split':
            for i in range(1, len(line)-1):
                set_glossary(trie, line, augmentations[1:], split_index=i)
        case 'bpe_dropout', dropout_rate, num_augmented:  # reference: https://arxiv.org/abs/1910.13267
            # TODO: implement using other tokenizers than tiktoken
            # BPE-dropout (https://github.com/VProv/BPE-Dropout/tree/master)
            # youtokentome (https://github.com/VKCOM/YouTokenToMe)
            # subword-nmt (https://github.com/rsennrich/subword-nmt)
            raise NotImplementedError
            set_glossary(trie, line, augmentations[1:])
            num_augmented = int(num_augmented)
            dropout_rate = float(dropout_rate)
            for _ in range(len(num_augmented)):
                set_glossary(trie, line[:i]+line[i+1:], augmentations[1:])
        case unknown:
            assert False, f"Invalid augmentation: {unknown}"
    
def get_bpe_trie(glossary_file,
                 visualize=False):
    trie = SubwordTrie()
    augmentations = []
    with open(glossary_file) as f:
        for line in f:
            line = line.strip()
            # skip empty lines and comments
            if line == "" or line.startswith("#"):
                continue

            # extract augmentation
            if line.startswith("@"):
                line = line[1:]
                augmentations.append(line)
                continue

            # extract complete bonujs
            if ':' in line:  # e.g., "Mestienne:0.01" means complete bonus of 0.01 for "Mestienne"
                assert line.count(':') == 1, f"Invalid line: {line}"
                line, complete_bonus = line.split(':')
                complete_bonus = float(complete_bonus)
            else:  # if no complete bonus is specified, use 0 (only continue bonus will be applied)
                complete_bonus = 0

            set_glossary(trie, line, augmentations=['first_capitalize', 'bpe_single_split'])

    if visualize:
        treevizer.to_png(trie.root, structure_type="trie", \
                         png_path=glossary_file+".png")
    return trie


if __name__ == "__main__":
    trie = get_bpe_trie(glossary_file=sys.argv[1], visualize=True)
    # breakpoint()