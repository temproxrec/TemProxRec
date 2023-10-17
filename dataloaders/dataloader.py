import torch
import torch.utils.data as data_utils
from .negative_samplers import negative_sampler_factory
import random

class Train_Dataset(data_utils.Dataset):
    
    """
    For train dataset & validation dataset
    - Task : MLM & MAM token masked sequence generation
    - Return : 20% masked item sequence / masked time sequence / label sequence
    """
    
    def __init__(self, u2seq, u2timeseq, max_len, mask_prob, mask_token, mask_time_token, num_items, num_time_items, rng):
        self.u2seq = u2seq
        self.u2timeseq = u2timeseq 
        self.users = sorted(self.u2seq.keys()) 
        self.max_len = max_len 
        self.mask_prob = mask_prob  
        self.mask_token = mask_token 
        self.mask_time_token = mask_time_token
        self.num_items = num_items 
        self.num_time_items = num_time_items
        self.rng = rng

    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, index):

        user = self.users[index]
        seq, timeseq = self._getseq(user)

        tokens = [] 
        labels = [] 

        if len(seq) == len(timeseq):
            for i in range(len(seq)):
                s = seq[i]
                prob = self.rng.random() 

                if prob < self.mask_prob:
                    prob /= self.mask_prob

                    if prob < 0.8:
                        tokens.append(self.mask_token)

                    elif prob < 0.9:
                        tokens.append(self.rng.randint(1, self.num_items))
                    else:
                        tokens.append(s)

                    labels.append(s)
                else:
                    tokens.append(s)
                    labels.append(0)
        
        else:
            print('error!')

        tokens = tokens[-self.max_len:]
        timeseq = timeseq[-self.max_len:]
        labels = labels[-self.max_len:] 

        mask_len = self.max_len - len(tokens)

        tokens = [0] * mask_len + tokens
        timeseq = [0] * mask_len + timeseq
        labels = [0] * mask_len + labels

        return  torch.LongTensor(tokens), torch.LongTensor(timeseq), torch.LongTensor(labels)
    
    def _getseq(self, user):
        return self.u2seq[user], self.u2timeseq[user]



class Val_Dataset(data_utils.Dataset): 

    def __init__(self, u2seq, u2timeseq, u2answer, u2time_answer, max_len, mask_token, negative_samples):
        self.u2seq = u2seq
        self.u2timeseq = u2timeseq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.u2time_answer = u2time_answer
        self.max_len = max_len
        self.mask_token = mask_token
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        timeseq = self.u2timeseq[user]
        answer = self.u2answer[user]
        time_answer = self.u2time_answer[user]
        negs = self.negative_samples[user]

        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)

        seq = seq + [self.mask_token]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
        answer = [0]*(self.max_len-1) + answer 
        
        timeseq = timeseq + time_answer
        timeseq = timeseq[-self.max_len:]
        timeseq = [0] * padding_len + timeseq
        
        return torch.LongTensor(seq), torch.LongTensor(timeseq), torch.LongTensor(answer), torch.LongTensor(candidates), torch.LongTensor(labels)



class Test_Dataset(data_utils.Dataset): 

    def __init__(self, u2seq, u2timeseq, u2seq_val, u2timeseq_val, u2answer, u2time_answer, max_len, mask_token, negative_samples):
        
        self.u2seq = u2seq
        self.u2timeseq = u2timeseq
        self.users = sorted(self.u2seq.keys())
        self.u2seq_val = u2seq_val
        self.u2timeseq_val = u2timeseq_val
        self.u2answer = u2answer
        self.u2time_answer = u2time_answer
        self.max_len = max_len
        self.mask_token = mask_token
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user] + self.u2seq_val[user]
        timeseq = self.u2timeseq[user] + self.u2timeseq_val[user]
        answer = self.u2answer[user]
        time_answer = self.u2time_answer[user]
        negs = self.negative_samples[user]

        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)

        seq = seq + [self.mask_token]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
        answer = [0]*(self.max_len-1) + answer ## 확인 필요
        
        timeseq = timeseq + time_answer
        timeseq = timeseq[-self.max_len:]
        timeseq = [0] * padding_len + timeseq
        
        return torch.LongTensor(seq), torch.LongTensor(timeseq), torch.LongTensor(answer), torch.LongTensor(candidates), torch.LongTensor(labels)



class Dataloader(): 
    
    """
    For train dataset & validation dataset loader
    - Task : MLM token masked data loader
    """

    def __init__(self, args, dataset):  
        
        self.args = args
        seed = args.seed
        self.rng = random.Random(seed)
        self.save_folder = dataset._get_preprocessed_folder_path()
        dataset = dataset.load_dataset()
        
        self.train = dataset['train']
        self.train_abstime = dataset['train_abstime']
        self.val = dataset['val']
        self.val_abstime = dataset['val_abstime']
        self.test = dataset['test']
        self.test_abstime = dataset['test_abstime']
        self.umap = dataset['umap']
        self.smap = dataset['smap']
        self.tmap = dataset['tmap']
        self.user_count = len(self.umap)
        self.item_count = len(self.smap)
        self.time_count = len(self.tmap)
        
        args.num_users = self.user_count
        args.num_items = self.item_count
        args.num_time_items = self.time_count
        
        self.max_len = args.maxlen 
        self.mask_prob = args.mask_prob 
        self.CLOZE_MASK_TOKEN = self.item_count + 1 
        self.CLOZE_MASK_TIME_TOKEN = self.time_count + 1
        
        code = args.test_negative_sampler_code
        test_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
                                                         self.user_count, self.item_count,
                                                         args.test_negative_sample_size,
                                                         args.seed,
                                                         self.save_folder)

        self.test_negative_samples = test_negative_sampler.get_negative_samples()
     
    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataloader = data_utils.DataLoader(self._get_train_dataset(), batch_size=self.args.batch_size,
                                           shuffle=False, pin_memory=True)
        return dataloader
    
    def _get_val_loader(self):
        dataloader = data_utils.DataLoader(self._get_val_dataset(), batch_size=self.args.batch_size,
                                           shuffle=False, pin_memory=True)
        return dataloader
    
    def _get_test_loader(self):
        dataloader = data_utils.DataLoader(self._get_test_dataset(), batch_size=self.args.batch_size,
                                           shuffle=False, pin_memory=True)
        return dataloader


    def _get_train_dataset(self):
        dataset = Train_Dataset(self.train, self.train_abstime, self.max_len, self.mask_prob, self.CLOZE_MASK_TOKEN, self.CLOZE_MASK_TIME_TOKEN, self.item_count, self.time_count, self.rng) 
        return dataset
    
    def _get_val_dataset(self):
        dataset = Val_Dataset(self.train, self.train_abstime, self.val, self.val_abstime, self.max_len, self.CLOZE_MASK_TOKEN, self.test_negative_samples)
        return dataset
        
    def _get_test_dataset(self):
        dataset = Test_Dataset(self.train, self.train_abstime, self.val, self.val_abstime, self.test, self.test_abstime, self.max_len, self.CLOZE_MASK_TOKEN, self.test_negative_samples)
        return dataset