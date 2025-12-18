import os
import regex as re
from collections import defaultdict, Counter
from collections.abc import Iterable, Iterator
import pickle
from concurrent.futures import ThreadPoolExecutor

from .pretokenization_example import find_chunk_boundaries

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    with open(input_path, "rb") as f:
        content = f.read()
    text = content.decode("utf-8", errors="ignore")
    if special_tokens:
        pattern = "|".join(re.escape(token) for token in special_tokens)
        segments = re.split(pattern, text)
    else:
        segments = [text]
    
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    word_freqs = Counter()
    for segment in segments:
        if segment:
            tokens = re.findall(PAT, segment)
            word_freqs.update(tokens)
            
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        token_bytes = token.encode('utf-8')
        if token_bytes not in vocab.values():
            vocab[len(vocab)] = token_bytes
    
    word_tokens = {}
    for word, freq in word_freqs.items():
        word_bytes = word.encode('utf-8')
        word_tokens[word] = [bytes([b]) for b in word_bytes]
    
    merges = []
    
    while len(vocab) < vocab_size:
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            tokens = word_tokens[word]
            if len(tokens) < 2:
                continue
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_freqs[pair] += freq
        if not pair_freqs:
            break

        best_pair = max(pair_freqs.items(), key=lambda x: (x[1], x[0]))[0]
        merges.append(best_pair)
        
        new_token = b"".join(best_pair)
        if new_token not in vocab.values():
            vocab[len(vocab)] = new_token
        
        words_to_update = []
        for word, tokens in word_tokens.items():
            if len(tokens) < 2:
                continue
            for i in range(len(tokens) - 1):
                if (tokens[i], tokens[i + 1]) == best_pair:
                    words_to_update.append(word)
                    break
        
        for word in words_to_update:
            tokens = word_tokens[word]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair):
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            word_tokens[word] = new_tokens
    
    return vocab, merges




class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        
        self.vocab_reverse = {token: token_id for token_id, token in vocab.items()}
        
        self.special_token_ids = {}
        if special_tokens:
            for token in special_tokens:
                token_bytes = token.encode('utf-8')
                if token_bytes in self.vocab_reverse:
                    self.special_token_ids[token] = self.vocab_reverse[token_bytes]
        
        self.merge_lookup = {}
        for pair in merges:
            self.merge_lookup[pair] = b"".join(pair)

        self.merge_priority = {pair: i for i, pair in enumerate(merges)}

    def encode(self, text: str) -> list[int]:
        result_ids = []
        if self.special_tokens:
            pattern = "|".join(re.escape(token) for token in self.special_tokens)
            segments = re.split(f"({pattern})", text)
            for segment in segments:
                if segment in self.special_tokens:
                    special_token_id = self.special_token_ids[segment]
                    if special_token_id is None:
                        result_ids.append(special_token_id)
                else:
                    result_ids.extend(self._encode_segment(segment))
        else:
            result_ids = self._encode_segment(text)
        return result_ids
    
    def _apply_bpe_merges(self, tokens: list[bytes]) -> list[bytes]:
        if len(tokens) <= 1:
            return tokens        
        while True:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            best_pair = None
            best_pos = -1
            best_merge_index = -1
            
            for i, pair in enumerate(pairs):
                if pair in self.merges:
                    merge_index = self.merge_priority[pair]
                    if best_merge_index == -1 or merge_index < best_merge_index:
                        best_pair = pair
                        best_pos = i
                        best_merge_index = merge_index
            
            if best_pair is None:
                break
            merged_token = self.merge_lookup[best_pair]
            tokens[best_pos] = merged_token
            tokens.pop(best_pos + 1)
        
        return tokens

    def _encode_segment(self, text: str) -> list[int]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pre_tokens = re.findall(PAT, text)
        
        result_ids = []
        for i, pre_token in enumerate(pre_tokens):
            pre_token_bytes = pre_token.encode('utf-8')
            tokens = [bytes([b]) for b in pre_token_bytes]

            tokens = self._apply_bpe_merges(tokens)
            
            for token in tokens:
                if token in self.vocab_reverse:
                    token_id = self.vocab_reverse[token]
                    result_ids.append(token_id)
                else:
                    for b in token:
                        byte_token = bytes([b])
                        token_id = self.vocab_reverse[byte_token]
                        result_ids.append(token_id)
        
        return result_ids
    

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
            yield from self._encode_file_in_chunks(iterable)
    
    def _encode_file_in_chunks(self, file_obj) -> Iterator[int]:
        num_processors = os.cpu_count() or 1
        boundaries = find_chunk_boundaries(file_obj, num_processors, b"<|endoftext|>")
        
        # 定义处理单个分块的函数
        def process_chunk(chunk_id, start, end):
            try:
                # 重新打开文件以避免线程冲突
                file_path = file_obj.name
                with open(file_path, 'rb') as chunk_file:
                    chunk_file.seek(start)
                    chunk_data = chunk_file.read(end - start)
                    
                    # 解码分块数据
                    try:
                        chunk_text = chunk_data.decode("utf-8", errors="ignore")
                    except UnicodeDecodeError:
                        chunk_text = chunk_data.decode("utf-8", errors="replace")
                    
                    # 编码这个分块
                    token_ids = self.encode(chunk_text)
                    
                    return chunk_id, token_ids
            except Exception as e:
                return chunk_id, [f"Error processing chunk {chunk_id}: {str(e)}"]
        
        # 使用线程池并行处理分块
        max_workers = num_processors
        chunk_tasks = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有分块任务
            for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
                future = executor.submit(process_chunk, i, start, end)
                chunk_tasks.append((i, future))
            
            # 收集结果并按顺序输出
            results = {}
            for chunk_id, future in chunk_tasks:
                try:
                    chunk_id, token_ids = future.result()
                    results[chunk_id] = token_ids
                except Exception as e:
                    results[chunk_id] = [f"Error in chunk {chunk_id}: {str(e)}"]
            
            # 按chunk_id顺序输出结果
            for i in range(len(results)):
                if i in results:
                    yield from results[i]

    def decode(self, ids: list[int]) -> str:
        text = [self.vocab[id].decode('utf-8', errors='ignore') for id in ids]
        return ''.join(text)

