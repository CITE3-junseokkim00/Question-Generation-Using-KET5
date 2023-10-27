import pandas as pd
import lightning.pytorch as pl
from transformers import T5ForConditionalGeneration,T5Tokenizer
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
import torchmetrics
import torch

class KET5QuestionGenerator(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained('KETI-AIR/ke-t5-base')
        self.tokenizer = T5Tokenizer.from_pretrained('KETI-AIR/ke-t5-base')
        self.lr = lr
        self.train_loss = []
        self.val_loss = []
        self.test_loss = []


    """
    return {
            'input_ids': np.array(input_ids, dtype=np.int_),
            'decoder_input_ids': np.array(dec_input_ids, dtype=np.int_),
            'label_ids': np.array(label_ids, dtype=np.int_)
        }
    """

    def forward(self, batch):
        attention_mask = batch['input_ids'].ne(self.tokenizer.pad_token_id).float()
        decoder_attention_mask = batch['decoder_input_ids'].ne(self.tokenizer.pad_token_id).float()
        return self.model(input_ids=batch['input_ids'], 
                          attention_mask=attention_mask, 
                          decoder_input_ids=batch['decoder_input_ids'],
                          decoder_attention_mask=decoder_attention_mask,
                          labels=batch['label_ids'], return_dict=True)
    
    def training_step(self, batch):
        output = self.forward(batch)
        self.train_loss.append(output.loss)
        self.log('train_loss', output.loss)
        return output.loss
    
    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        self.train_loss.append(output.loss)
        self.log('train_loss', output.loss)
        return output.loss
    
    def test_step(self, batch, batch_idx):
        output = self.forward(batch)
        self.train_loss.append(output.loss)
        self.log('train_loss', output.loss)
        return output.loss
    
    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr, correct_bias=False)

        # scheduler
        scheduler = get_cosine_schedule_with_warmup(optimizer,
            num_warmup_steps=int(self.trainer.estimated_stepping_batches * 0.1),
            num_training_steps=self.trainer.estimated_stepping_batches,)
        
        lr_scheduler = {
            'scheduler': scheduler,
            'monitor': 'val_loss',
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [lr_scheduler]