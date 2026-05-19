import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience: 
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

class Metrics:
    def __init__(self, class_names):
        self.class_names = class_names

    def calculate_metrics(self, y_true, y_pred, y_probs=None):
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        metrics = {'accuracy': acc, 'f1_macro': f1}
        
        if y_probs is not None:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
            except:
                metrics['auc'] = 0.0

        cm = confusion_matrix(y_true, y_pred, labels=range(len(self.class_names)))
        metrics['cm_data'] = cm # Store raw matrix for the plotter
        
        for i, name in enumerate(self.class_names):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - (tp + fn + fp)
            metrics[f"{name.lower()}_sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics[f"{name.lower()}_specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
        return metrics

class Trainer:
    def __init__(self, model, device, class_names, model_name='customcnn', aug_type='standard', class_weights = None):
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names
        self.aug_type = aug_type
        self.model_name = f'{model_name}_{aug_type}'
        self.class_weights = class_weights
        self.metrics_calc = Metrics(class_names)
        
        self.base_dir = os.path.join('Result', self.model_name)
        self.ckpt_dir = os.path.join(self.base_dir, 'checkpoints')
        self.logs_dir = os.path.join(self.base_dir, 'logs')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        self.history = {
            'train_accuracy': [],
            'train_loss': [],
            'val_accuracy': [],
            'val_loss': [],
            'val_auc': [],
            'train_auc': [],
        }

    def train(self, train_loader, val_loader, epochs=50, criterion=None, lr=0.001):
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        
        # ROAD DAMAGE FIX: Apply class weights to loss function if provided
        if criterion is None:
            if self.class_weights is not None:
                criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))
            else:
                criterion = nn.CrossEntropyLoss()
                
        early_stop = EarlyStopping(patience=10)
        best_acc = 0.0

        for epoch in range(epochs):
            self.model.train()
            epoch_train_loss = 0.0
            all_train_preds, all_train_labels = [], []
            
            for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_train_preds.extend(preds.cpu().numpy())
                all_train_labels.extend(labels.cpu().numpy())

            avg_train_loss = epoch_train_loss / len(train_loader)
            train_acc = accuracy_score(all_train_labels, all_train_preds)
            
            # Validation call (we don't save CM every epoch to save time)
            val_results, val_loss = self.evaluate(val_loader, criterion, save_cm=False)
            
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_accuracy'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_results['accuracy'])
            self.history['val_auc'].append(val_results.get('auc', 0.0))

            logger.info(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_results['accuracy']:.4f}")
            logger.info(f"           Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}, Val AUC={val_results.get('auc', 0.0):.4f}")

            if val_results['accuracy'] > best_acc:
                best_acc = val_results['accuracy']
                torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, 'best_model.pth'))

            if early_stop(val_loss):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        self.save_final_metrics()
        self.plot_history()

    def save_final_metrics(self):
        final_path = os.path.join(self.base_dir, 'final_trainMetrics.json')
        self.history['aug_type'] = self.aug_type
        self.history['model_name'] = self.model_name
        with open(final_path, 'w') as f:
            json.dump(self.history, f, indent=4)

    def plot_history(self):
        # Dynamically get the actual number of epochs (to handle early stopping)
        actual_epochs = len(self.history['train_accuracy'])
        epochs_range = range(actual_epochs)

        # Use one figure wide enough to hold both subplots side-by-side
        plt.figure(figsize=(16, 5))
        
        # First Plot: Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, self.history['train_accuracy'], label="Trainnign Accuracy")
        plt.plot(epochs_range, self.history['val_accuracy'], label="Validation Accuracy")
        plt.legend(loc='lower right')
        plt.title("Trainnign & Validation Accuracy")

        # Second Plot: Loss
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, self.history['train_loss'], label="Trainnign Loss")
        plt.plot(epochs_range, self.history['val_loss'], label="Validation Loss")
        plt.legend(loc='upper right')
        plt.title("Trainnign & Validation Loss")
        
        plt.tight_layout()
        # Save before show so the file isn't empty
        plt.savefig(os.path.join(self.base_dir, 'training_curves.png'))
        #plt.show()
        plt.close()

    def save_confusion_matrix(self, cm, filename='confusion_matrix.png', prefix=''):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title(f'{prefix} Confusion Matrix - {self.model_name}')
        
        # Prevent Validation and Test runs from overwriting each other
        save_name = f"{prefix.lower()}_{filename}" if prefix else filename
        plt.savefig(os.path.join(self.base_dir, save_name))
        plt.close()

    def evaluate(self, loader, criterion=None, save_cm=True, prefix='Val'):
        self.model.eval()
        all_preds, all_labels, all_probs, total_loss = [], [], [], 0.0
        
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                if criterion:
                    total_loss += criterion(outputs, labels).item()
                
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        metrics = self.metrics_calc.calculate_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
        
        if save_cm:
            self.save_confusion_matrix(metrics['cm_data'], prefix=prefix)
            
        avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
        return metrics, avg_loss