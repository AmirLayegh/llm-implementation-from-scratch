import torch 
from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
            
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
# a dataloader that creates batches of data
def create_dataloader_v1(txt, tokenizer, max_length=4, stride=4, batch_size=8, shuffle=False, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return data_loader

def create_embedding(vocab_size, output_dim=4):
    embedding = torch.nn.Embedding(vocab_size, output_dim)
    return embedding
    


if __name__ == "__main__":
    with open("verdict.txt", "r", encoding="utf-8") as f:
        txt = f.read()
    tokenizer = tiktoken.get_encoding("gpt2")
    
    dataloader = create_dataloader_v1(txt, tokenizer=tokenizer,batch_size=8, max_length=4, stride=4, shuffle=False)
    data_iter = iter(dataloader)
    # first_batch = next(data_iter)
    # print(type(first_batch))
    # print(first_batch)
    # print(first_batch[0].shape, first_batch[1].shape)
    # print(first_batch[0].shape)
    # input_ids = first_batch[0][0]
    # print(input_ids)
    # print(f"The input tensor: {tokenizer.decode(first_batch[0].tolist()[0])}")
    # print(f"The target tensor: {tokenizer.decode(first_batch[1].tolist()[0])}")
    inputs, targets = next(data_iter)
    print(f"Token IDs: {inputs}")
    print(f"Inpust shapes: {inputs.shape}")
    
    vocab_size = 50257
    output_dim = 256
    embedding_layer = create_embedding(vocab_size, output_dim)
    token_embeddings = embedding_layer(inputs)
    print(f"Token embedding shapes: {token_embeddings.shape}")
    
    context_length = 4
    pos_embedding_layer = create_embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    print(f"Pos embedding shapes: {pos_embeddings.shape}")
    
    input_embeddings = token_embeddings + pos_embeddings
    print(f"Input embeddings shapes: {input_embeddings.shape}")
    
    #print("Embedding: ", embedding_layer(input_ids))
    
    # second_batch = next(data_iter)
    # print(second_batch)
    # print(f"The input tensor: {tokenizer.decode(second_batch[0].tolist()[0])}")
    # print(f"The target tensor: {tokenizer.decode(second_batch[1].tolist()[0])}")
    
    # inputs, targets = second_batch
    # print("Inputs: ", inputs)
    # print("Targets: ", targets)
    
    # input_ids = torch.tensor([1, 2, 3, 4])
    # vocab_size = tokenizer.n_vocab
    # print("Vocab size: ", vocab_size)
    # output_dim = 4
    # embedding = create_embedding(input_ids, input_ids, vocab_size, output_dim)
    # print("Inputs: ", embedding)
    #print("Targets: ", targets.shape)
    
    
    
    