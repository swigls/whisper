from typing import Optional, List, Tuple, Iterable
import numpy as np

import whisper
from whisper.tokenizer import get_tokenizer
from whisper.normalizers import EnglishTextNormalizer

import os
import sys
from dataclasses import dataclass 


class SingletonTokenizer:
    # Whisper tokenizer, but singleton
    # (to avoid loading the tokenizer multiple times)
    tokenizer = None
    @classmethod
    def encode(cls, text: str) -> List[int]:
        return cls.get_tokenizer().encode(text)
    @classmethod
    def decode(cls, tokens: List[int]) -> str:
        return cls.get_tokenizer().decode(tokens)
    @classmethod
    def get_tokenizer(cls):
        if cls.tokenizer is None:
            assert False, "SingletonTokenizer not initialized"
            cls.tokenizer = get_tokenizer()
        return cls.tokenizer
    @classmethod
    def init_tokenizer(cls, tokenizer):
        cls.tokenizer = tokenizer

class SubwordTrieNode:
    def __init__(self, subword: Optional[int]=None):
        self.subword = subword
        self.stop: bool = False
        self.children: List[SubwordTrieNode] = []

        self.complete_bonus: float = 0
    @property
    def value(self):
        if self.subword is None:
            return ''
        else:
            s = SingletonTokenizer.decode([self.subword])         
            return f'_{s}' if s[0] != ' ' else s

class SubwordTrie:
    def __init__(self):
        self.root = SubwordTrieNode()
    def set(self,
            subwords: Tuple[int],
            complete_bonus: float = 0,
            ):
        node = self.root
        for subword in subwords:
            if not node.children:
                node.children.append(SubwordTrieNode(subword))
                node = node.children[-1]
            else:
                for child in node.children:
                    if child.subword == subword:
                        node = child
                        break
                else:
                    node.children.append(SubwordTrieNode(subword))
                    node = node.children[-1]
        node.stop = True
        node.complete_bonus = complete_bonus
    def traverse(self,
                 subwords: Tuple[int],
                 node: Optional[SubwordTrieNode] = None,
                 ) -> Optional[SubwordTrieNode]:
        if node is None:
            node = self.root
        for subword in subwords:
            if not node.children:
                return None
            else:
                for child in node.children:
                    if child.subword == subword:
                        node = child
                        break
                else:
                    return None
        return node