import torch
import torch.nn as nn
from torch.optim import Adam

from tqdm import tqdm
from tensorboardX import SummaryWriter


#from models.recommender import Recommender
from loggers import *
from .utils import *
from .tcl import *
from models.modules.encoder import Encoder
from models.model import Model

class Trainer:
    """

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Temporal-Proximity-Aware Contrastive Learning : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """
        
    def __init__(self, encoder: Encoder, args, train_dataloader, val_dataloader, test_dataloader,  export_root,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10):

        # Setup cuda device for  training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        
        self.args = args
        self.dataset_code = args.dataset_code

        self.num_items = args.num_items
        self.num_time_items = args.num_time_items 
        self.tcl = args.tcl
        self.device = torch.device("cuda:"+args.device_idx if cuda_condition else "cpu")
        
        # Define Model
        self.encoder = encoder
        self.model = Model(encoder).to(self.device)
        for name, param in self.model.named_parameters():
            try:
                torch.nn.init.xavier_normal_(param.data)
            except:
                pass 
        
        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.valid_data = val_dataloader
        self.test_data = test_dataloader
        
        # Setting the Adam optimizer with hyper-param
        self.optimizer = self._create_optimizer()
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.decay_step, gamma=args.gamma)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        self.metric_ks = args.metric_ks
        
        self.num_epochs = args.num_epochs
        self.best_metric = args.best_metric

        self.export_root = export_root
        self.writer, self.train_loggers, self.val_loggers = self._create_loggers()
        self.add_extra_loggers()
        self.logger_service = LoggerService(self.train_loggers, self.val_loggers)
        self.log_period_as_iter = args.log_period_as_iter
        
        self.criterion = nn.NLLLoss(ignore_index=0)
        self.log_freq = log_freq
    
    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def log_extra_val_info(self, log_data):
        pass
    
    
    def train(self):
        
        accum_iter = 0
        self.validate(0, accum_iter)
        for epoch in range(self.num_epochs):
            accum_iter = self.train_one_epoch(epoch, accum_iter)
            self.validate(epoch, accum_iter)
        self.logger_service.complete({
            'state_dict': (self._create_state_dict()),
        })
        self.writer.close()


    
    def train_one_epoch(self, epoch, accum_iter, ctl_all = True): 
        self.model.train()
        
        average_meter_set = AverageMeterSet()
        tqdm_dataloader = tqdm(self.train_data)

                             
        for i, data in enumerate(tqdm_dataloader):

            batch_size = data[0].size(0)
            data = [x.to(self.device) for x in data]
            
            # 2-1. NLLLoss(negative log likelihood) of predicting masked token 
            self.optimizer.zero_grad()
            mask_loss, hidden, aug_hidden = self.calculate_loss(data)
            
            #2-2. Contrastive Learning Loss time aware Contrastive learning loss            
            if self.args.tcl == 'no':
                loss = mask_loss
                average_meter_set.update('tcl_loss', 0)
           
            else:
                times = data[1]
                tcl_loss = self.process_data(hidden, aug_hidden, times, self.args, batch_size, self.device)
                loss = mask_loss + self.args.lamb*tcl_loss
                average_meter_set.update('tcl_loss', tcl_loss.item())
                
            loss.backward()
            self.optimizer.step()  
 
            average_meter_set.update('loss', loss.item())
            average_meter_set.update('mask_loss', mask_loss.item())
            
            tqdm_dataloader.set_description(
                'Epoch {}, loss {:.3f} | mask_loss {:.3f} | tcl_loss {:.3f} '.format(epoch+1, average_meter_set['loss'].avg, average_meter_set['mask_loss'].avg, average_meter_set['tcl_loss'].avg))

            accum_iter += batch_size

            if self._needs_to_log(accum_iter):
                tqdm_dataloader.set_description('Logging to Tensorboard')
                log_data = {
                    'state_dict': (self._create_state_dict()),
                    'epoch': epoch,
                    'accum_iter': accum_iter,
                }
                log_data.update(average_meter_set.averages())
                self.log_extra_train_info(log_data)
                self.logger_service.log_train(log_data)
        self.lr_scheduler.step()
        return accum_iter
    
    
    def process_data(self, hidden, aug_hidden, times, args, batch_size, device): 
    
        qry_idxs = [len(time_seq) - 1 for time_seq in times] #[batch_size]
        query_times = [time_seq[-1] for time_seq in times] #[batch_size]

        query_hids = torch.stack([hidden[seq_idx][qry_idx] for seq_idx, qry_idx in enumerate(qry_idxs)]) #[batch_size, hidden]
        aug_query_hids = torch.stack([aug_hidden[seq_idx][qry_idx] for seq_idx, qry_idx in enumerate(qry_idxs)]) #[batch_size, hidden]
        
        new_seq_hiddens = [torch.cat((hidden[seq_idx][:qry_idx], aug_query_hid.unsqueeze(0)), dim=0) for
                            seq_idx, (qry_idx, aug_query_hid) in enumerate(zip(qry_idxs, aug_query_hids))] #각 new_seq_hiddens : [max len, hidden]

        new_batch_hidden = torch.stack(
            [torch.cat((hidden[:seq_idx], seq_hidden.unsqueeze(0), hidden[seq_idx + 1:]), dim=0) for
                seq_idx, seq_hidden in enumerate(new_seq_hiddens)]) # [qry num, batch_size, max len, hidden]

        query_times = torch.stack(query_times) # [batch_size]
        tcl_loss = TCL(query_hids, new_batch_hidden, query_times, times, args.num_time_items, args, batch_size, device)
 
        return tcl_loss
   
   
    def validate(self, epoch, accum_iter):
        
        self.model.eval()
        average_meter_set = AverageMeterSet()

        with torch.no_grad():

            tqdm_dataloader = tqdm(self.valid_data)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]
                metrics = self.calculate_metrics(batch)

                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] +\
                                      ['Recall@%d' % k for k in self.metric_ks[:3]]
                description = 'Val: ' + ', '.join(s + ' {:.3f}' for s in description_metrics)
                description = description.replace('NDCG', 'N').replace('Recall', 'R')
                description = description.format(*(average_meter_set[k].avg for k in description_metrics))
                tqdm_dataloader.set_description(description)
                
            log_data = {
                'state_dict': (self._create_state_dict()),
                'epoch': epoch,
                'accum_iter': accum_iter,
            }
            log_data.update(average_meter_set.averages())
            self.logger_service.log_val(log_data)
            
        return average_meter_set


    def test(self):

        best_model = torch.load(os.path.join(self.export_root, 'models', 'best_acc_model.pth')).get('model_state_dict')
        self.model.load_state_dict(best_model)
        self.model.eval()

        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_data)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]

                metrics = self.calculate_metrics(batch)

                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] +\
                                      ['Recall@%d' % k for k in self.metric_ks[:3]]
                description = 'Val: ' + ', '.join(s + ' {:.3f}' for s in description_metrics)
                description = description.replace('NDCG', 'N').replace('Recall', 'R')
                description = description.format(*(average_meter_set[k].avg for k in description_metrics))
                tqdm_dataloader.set_description(description)

            average_metrics = average_meter_set.averages()
            with open(os.path.join(self.export_root, 'logs', 'test_metrics.json'), 'w') as f:
                json.dump(average_metrics, f, indent=4)
            
            print(average_metrics)


    def calculate_loss(self, batch):
        
        seqs, tseps, answers = batch[0], batch[1], batch[2]
        
        out, hidden = self.model(seqs, tseps)  # B x T x V
        _, hidden2 = self.model(seqs, tseps) 
        logits = out.view(-1, out.size(-1))  # (B*T) x V
        answers = answers.view(-1)  # B*T
        loss = self.ce(logits, answers)
        
        return loss, hidden, hidden2


    def calculate_metrics(self, batch):

        seqs, tseps, candidates, labels = batch[0], batch[1], batch[3],  batch[4]
        
        out, _ = self.model(seqs, tseps)
        scores = out[:, -1, :]  # B x V
        scores = scores.gather(1, candidates)  # B x C
        metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        
        return metrics
    
    
    def _create_loggers(self):
        root = Path(self.export_root)
        writer = SummaryWriter(root.joinpath('logs'))
        model_checkpoint = root.joinpath('models')

        train_loggers = [
            MetricGraphPrinter(writer, key='epoch', graph_name='Epoch', group_name='Train'),
            MetricGraphPrinter(writer, key='loss', graph_name='Loss', group_name='Train'),
        ]

        val_loggers = []
        for k in self.metric_ks:
            val_loggers.append(
                MetricGraphPrinter(writer, key='NDCG@%d' % k, graph_name='NDCG@%d' % k, group_name='Validation'))
            val_loggers.append(
                MetricGraphPrinter(writer, key='Recall@%d' % k, graph_name='Recall@%d' % k, group_name='Validation'))
        val_loggers.append(RecentModelLogger(model_checkpoint))
        val_loggers.append(BestModelLogger(model_checkpoint, metric_key=self.best_metric))
        return writer, train_loggers, val_loggers


    def _create_state_dict(self):
        return {
            STATE_DICT_KEY: self.model.state_dict(),
            OPTIMIZER_STATE_DICT_KEY: self.optimizer.state_dict(),
        }


    def _needs_to_log(self, accum_iter):
        return accum_iter % self.log_period_as_iter < self.args.batch_size and accum_iter != 0
    
    
    def _create_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)