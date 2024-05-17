import torch

SOW_token = 0
EOW_token = 1

class CharEncoder:
    def __init__(self, name, device):
        self.name = name
	#Dictionary mapping characters to index
        self.char2index = {}
	#Dictionary mapping index to characters
        self.index2char = {SOW_token: "<", EOW_token: ">"}
        self.n_chars = 2  # Count SOW and EOW
        self.device = device

    def encode(self, word):
	#Update the dictionaries created for mapping
        for char in word:
            if char not in self.char2index:
                self.char2index[char] = self.n_chars
                self.index2char[self.n_chars] = char
                self.n_chars += 1
                
    def getTensorFromWord(self, word:str):
	#Given a word create a tensor with respective char indices and append EOW_token at the end
        indexes =[]
        for char in word:
            if(self.char2index.__contains__(char)):
                indexes.append(self.char2index[char])
        indexes.append(EOW_token)
        return torch.tensor(indexes, dtype=torch.long, device=self.device).view(-1, 1)
    
        