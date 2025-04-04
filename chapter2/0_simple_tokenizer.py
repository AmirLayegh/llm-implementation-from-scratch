import urllib.request
import re

url = ("https://raw.githubusercontent.com/rasbt/"
"LLMs-from-scratch/main/ch02/01_main-chapter-code/"
"the-verdict.txt")

file_path = "verdict.txt"
urllib.request.urlretrieve(url, file_path)

with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()
    
#print("Total number of characters: ", len(raw_text))

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

#print("Total number of preprocessed tokens: ", len(preprocessed))
#print(preprocessed[:100])

all_words = sorted(set(preprocessed))
# extend the vocabulary with special tokens
all_words.extend(["<|endoftext|>", "<|unk|>"])
vocab_size = len(all_words)
#print("Vocabulary size: ", vocab_size)

vocab = {token:integer for integer, token in enumerate(all_words)}

# for i, item in enumerate(vocab.items()):
#     print(f"{i}: {item}")
#     if i>=30:
#         break

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {v:k for k,v in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        token_ids = [self.str_to_int[token] for token in preprocessed]
        return token_ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[id] for id in ids])
        text = text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        
        return text
    
tokenizer = SimpleTokenizerV1(vocab)
text = """It's the last he painted, you know,"
        Mrs. Gisburn said with a pride."""
        
ids = tokenizer.encode(text)
print(ids)

decoded_text = tokenizer.decode(ids)
print(decoded_text)


class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {v:k for k,v in vocab.items()}
        
        
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int 
                        else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[item] for item in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[id] for id in ids])
        text = text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
    
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)

tokenizer = SimpleTokenizerV2(vocab)
ids = tokenizer.encode(text)
print(ids)

decoded_text = tokenizer.decode(ids)
print(decoded_text)
    
    
        