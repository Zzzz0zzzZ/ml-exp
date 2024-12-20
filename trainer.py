import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np

class Trainer:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.device = args.device
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        
    def train(self, train_dataloader, val_dataloader):
        num_training_steps = len(train_dataloader) * self.args.epochs
        num_warmup_steps = int(num_training_steps * self.args.warmup_ratio)
        
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        best_val_acc = 0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(self.args.epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_acc = 0
            
            train_pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{self.args.epochs} [Train]')
            for batch in train_pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                train_acc += (preds == labels).sum().item()
                
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            train_loss /= len(train_dataloader)
            train_acc /= len(train_dataloader.dataset)
            
            # Validation
            val_loss, val_acc = self.evaluate(val_dataloader)
            
            print(f'Epoch {epoch+1}/{self.args.epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.args.early_stopping_patience:
                print(f'Early stopping triggered after epoch {epoch+1}')
                break
        
        # Load best model
        self.model.load_state_dict(best_model_state)
        return best_val_acc
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        total_acc = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                total_acc += (preds == labels).sum().item()
        
        return total_loss / len(dataloader), total_acc / len(dataloader.dataset)
    
    def predict(self, test_dataloader):
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc='Predicting'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                preds = torch.argmax(outputs, dim=1)
                predictions.extend(preds.cpu().numpy())
                
        return predictions 