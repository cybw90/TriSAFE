import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split
from sklearn.model_selection import KFold
from typing import Dict, List, Optional, Tuple, Any
import logging
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field
from attack_trainer import AttackTrainer
from config import GlobalConfig
from data_loader import create_federated_data_loaders
from phe_mechanism import EncryptedPackedValue
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from server_ops import ServerOperations, ServerConfig

# Import WorkerOperations - make it required, not optional
from worker_ops import WorkerOperations, create_worker, WorkerConfig


@dataclass
class ModelConfig:
    """Lightweight architecture config for TinyMLNetwork.

    Defined here (was previously imported from config.py, where it does not
    exist). Provides the two attributes TinyMLNetwork reads.
    """
    hidden_dims: list = field(default_factory=lambda: [64, 32])
    dropout_rate: float = 0.1

class TinyMLNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, config: ModelConfig = None):
        super(TinyMLNetwork, self).__init__()
        if config is None:
            config = ModelConfig()
        
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.dropout_rate = float(config.dropout_rate)
        
        if hasattr(config, 'hidden_dims') and config.hidden_dims:
            self.hidden_dims = [int(dim) for dim in config.hidden_dims]
        else:
            self.hidden_dims = [64, 32]
        
        self.input_norm = nn.LayerNorm(self.input_dim)
        
        layers = []
        prev_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(self.hidden_dims[-1], self.output_dim)
        
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Expected input dimension of {self.input_dim}, but got {x.shape[1]}")
        
        x = self.quant(x)
        x = self.input_norm(x)
        x = self.feature_extractor(x)
        x = self.classifier(x)
        x = self.dequant(x)
        
        return x

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
            
        x = self.quant(x)
        x = self.input_norm(x)
        x = self.feature_extractor(x)
        return x

class ModelTrainer:
    """
    ModelTrainer that properly integrates with WorkerOperations for TriSAFE protocol
    """
    def __init__(
        self,
        model: nn.Module,
        config: WorkerConfig,
        worker_ops: Optional[WorkerOperations] = None
    ):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.metrics_history = defaultdict(list)
        
        # REQUIRED: WorkerOperations for TriSAFE protocol
        self.worker_ops = worker_ops
        
        # Training parameters
        self.n_folds = config.trainer.n_folds
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        
        # Early stopping
        self.patience = config.trainer.patience
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer setup
        if hasattr(config, 'optimizer_type') and config.optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=config.trainer.learning_rate,
                momentum=config.momentum,
                weight_decay=config.trainer.weight_decay
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=config.trainer.learning_rate,
                weight_decay=config.trainer.weight_decay
            )
        
        # LR scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.trainer.scheduler_factor,
            patience=config.trainer.scheduler_patience,
            min_lr=config.trainer.scheduler_min_lr
        )
        
        self.max_grad_norm = config.trainer.max_grad_norm
        self.logger = logging.getLogger(f'ModelTrainer-{config.worker_id}')


    def _train_fold(self, train_loader: Optional[DataLoader], val_loader: DataLoader, attack_trainer=None) -> Dict[str, float]:
        """Training fold with proper error handling"""
        try:
            best_val_loss = float('inf')
            best_metrics = None
            patience_counter = 0
            
            for epoch in range(self.epochs):
                # Training phase
                train_metrics = self.train_epoch(train_loader, attack_trainer)
                if train_metrics is None or not isinstance(train_metrics, dict):
                    train_metrics = {'loss': float('inf'), 'accuracy': 0.0}

                # Evaluation phase
                val_metrics = self.evaluate(val_loader)
                if val_metrics is None or not isinstance(val_metrics, dict):
                    val_metrics = {'loss': float('inf'), 'accuracy': 0.0}
                
                # Handle validation improvement
                current_val_loss = val_metrics.get('loss', float('inf'))
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    best_metrics = {
                        'train_loss': train_metrics.get('loss', float('inf')),
                        'train_accuracy': train_metrics.get('accuracy', 0.0),
                        'val_loss': val_metrics.get('loss', float('inf')),
                        'val_accuracy': val_metrics.get('accuracy', 0.0)
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping check
                if patience_counter >= self.patience:
                    break
                
                # Update learning rate scheduler
                self.scheduler.step(current_val_loss)
                
            return best_metrics if best_metrics is not None else {
                'train_loss': float('inf'),
                'train_accuracy': 0.0,
                'val_loss': float('inf'),
                'val_accuracy': 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error in fold training: {str(e)}")
            return {
                'train_loss': float('inf'),
                'train_accuracy': 0.0,
                'val_loss': float('inf'),
                'val_accuracy': 0.0
            }

    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluation method"""
        self.model.eval()
        metrics = defaultdict(float)
        num_samples = 0
        
        try:
            with torch.no_grad():
                for batch_data, batch_labels in data_loader:
                    batch_data = torch.nan_to_num(batch_data, nan=0.0, posinf=1e6, neginf=-1e6).to(self.device)
                    batch_labels = batch_labels.long().to(self.device)
                    
                    outputs = self.model(batch_data)
                    loss = self.criterion(outputs, batch_labels)
                    
                    predictions = torch.argmax(outputs, dim=1)
                    accuracy = (predictions == batch_labels).float().mean().item()
                    
                    batch_size = batch_labels.size(0)
                    metrics['loss'] = (
                        metrics['loss'] * num_samples + loss.item() * batch_size
                    ) / (num_samples + batch_size)
                    metrics['accuracy'] = (
                        metrics['accuracy'] * num_samples + accuracy * batch_size
                    ) / (num_samples + batch_size)
                    num_samples += batch_size
            
            return self._finalize_metrics(metrics)
        except Exception as e:
            self.logger.error(f"Evaluation error: {str(e)}")
            return {'loss': float('inf'), 'accuracy': 0.0}


    def train_epoch(self, data_loader: Optional[DataLoader] = None, attack_trainer=None) -> Dict[str, float]:
        """
        Training epoch that properly uses WorkerOperations for TriSAFE protocol
        """
        self.model.train()
        running_loss = 0.0
        running_acc = 0.0
        num_samples = 0
        total_attack_impact = 0.0
        
        # Use WorkerOperations if available for proper TriSAFE protocol
        if self.worker_ops:
            # Use WorkerOperations' train_local_model for proper cryptographic processing
            gradients, metadata = self.worker_ops.train_local_model()
            
            # Extract metrics from metadata
            if metadata and 'performance' in metadata:
                return {
                    'loss': metadata['performance'].get('loss', 0.0),
                    'accuracy': metadata['performance'].get('accuracy', 0.0),
                    'has_bulletproof': 'bulletproof' in metadata,
                    'has_pep_proof': 'pep_proof' in metadata
                }
        
        # Fallback to original implementation if no WorkerOperations
        loader = data_loader
        if not loader:
            self.logger.warning("No data_loader provided, skipping.")
            return {'loss': float('inf'), 'accuracy': 0.0}

        try:
            # Handle pre-batch attack
            if attack_trainer and attack_trainer.is_malicious and attack_trainer.attack_type == 'label_flip':
                attack_trainer._ensure_label_flip()

            for batch_idx, (batch_data, batch_labels) in enumerate(loader):
                current_batch_size = batch_data.size(0)
                
                batch_data = torch.nan_to_num(batch_data, nan=0.0, posinf=1e6, neginf=-1e6).to(self.device)
                batch_labels = batch_labels.long().to(self.device)
                
                self.optimizer.zero_grad(set_to_none=True)
                
                try:
                    outputs = self.model(batch_data)
                    loss = self.criterion(outputs, batch_labels)
                    
                    if torch.isfinite(loss) and loss.item() < 1e6:
                        loss.backward()
                        
                        # Handle attacks and gradients
                        attack_impact = 0.0
                        if attack_trainer and attack_trainer.is_malicious:
                            original_grads = {
                                name: param.grad.clone()
                                for name, param in self.model.named_parameters()
                                if param.grad is not None
                            }
                            malicious_grads, attack_metrics = attack_trainer.get_malicious_updates(original_grads)
                            if malicious_grads:
                                for name, param in self.model.named_parameters():
                                    if name in malicious_grads:
                                        param.grad.data.copy_(malicious_grads[name])
                                attack_impact = attack_metrics.get('attack_impact', 0.0)

                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.optimizer.step()

                        # Update metrics
                        with torch.no_grad():
                            preds = torch.argmax(outputs, dim=1)
                            accuracy = (preds == batch_labels).float().mean().item()
                            
                            running_loss += loss.item() * current_batch_size
                            running_acc += accuracy * current_batch_size
                            num_samples += current_batch_size
                            total_attack_impact += attack_impact
                    else:
                        self.logger.warning(f"Skipping batch {batch_idx+1} due to invalid loss: {loss.item()}")
                
                except RuntimeError as e:
                    self.logger.warning(f"Error processing batch {batch_idx}: {str(e)}")
                    continue

            # Return metrics
            if num_samples > 0:
                metrics = {
                    'loss': running_loss / num_samples,
                    'accuracy': running_acc / num_samples
                }
                if attack_trainer and attack_trainer.is_malicious:
                    metrics['attack_impact'] = total_attack_impact / len(loader)
                
                return self._finalize_metrics(metrics)
            else:
                self.logger.warning("No samples processed in epoch")
                return {'loss': float('inf'), 'accuracy': 0.0}

        except Exception as e:
            self.logger.error(f"Training epoch error: {str(e)}")
            return {'loss': float('inf'), 'accuracy': 0.0}

    # Keep other methods unchanged but add this helper
    def get_gradients_for_trisafe(self) -> List[torch.Tensor]:
        """Extract gradients for TriSAFE protocol processing"""
        gradients = []
        for param in self.model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.clone())
            else:
                gradients.append(torch.zeros_like(param))
        return gradients

    # Rest of the methods remain the same...
    # [Include all other unchanged methods from original]

class FederatedTrainer:
    def __init__(self, server_config: ServerConfig, global_config: GlobalConfig, existing_loaders=None):
        self.server_config = server_config
        self.global_config = global_config
        self.batch_buffer = []
        self.current_round = 0

        if not hasattr(self.global_config, 'epochs'):
            self.global_config.epochs = 10

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use dynamic output_dim from config, not hardcoded
        self.model = TinyMLNetwork(
            input_dim=global_config.input_dim,
            output_dim=global_config.output_dim,
            config=global_config.model
        ).to(self.device)
        
        # Initialize ServerOperations with proper model
        self.server_ops = ServerOperations(
            config=server_config,
            model=self.model
        )
        
        self.worker_trainers = []
        self.logger = logging.getLogger('FederatedTrainer')
        self.round_metrics = defaultdict(list)
        
        # Data loaders
        if existing_loaders is not None:
            self.worker_loaders, self.val_loader, self.test_loader = existing_loaders
        else:
            self.worker_loaders = None
            self.val_loader = None
            self.test_loader = None
        
        self.client_metrics = defaultdict(lambda: {
            'accuracy_history': [],
            'loss_history': [],
            'val_loss_history': [],
            'val_accuracy_history': [],
            'learning_rates': []
        })
        
        self.best_global_val_loss = float('inf')
        self.best_global_model_state = None

    def update_batch_buffer(self, worker_id: int, batch, proof: Dict):
        """Update batch buffer with new proof including time window validation"""
        current_time = time.time()
        
        # Preserve manipulated timestamps for time delay attacks
        if 'timestamp' not in proof:
            proof['timestamp'] = current_time
            
        proof['worker_id'] = worker_id
        
        # Add new batch with proof
        self.batch_buffer.append([(batch, proof)])
        
        # Keep only recent batches (within 2x time window - should be 600s not 20s!)
        max_age = 2 * self.server_config.time_window
        self.batch_buffer = [
            batch for batch in self.batch_buffer
            if current_time - batch[0][1].get('timestamp', 0) <= max_age
        ]

    def initialize_workers(self, num_workers: Optional[int] = None):
        """Initialize workers with proper TriSAFE WorkerOperations"""
        try:
            n_workers = num_workers if num_workers is not None else self.global_config.num_workers
            
            # Clear existing workers
            self.worker_trainers = []
            self.workers = {}
            self.attack_trainers = {}
            self.worker_operations = {}  # Store WorkerOperations instances

            for i in range(n_workers):
                worker_config = WorkerConfig(
                    global_config=self.global_config,
                    worker_id=i
                )
                
                # Clone the global model
                model = TinyMLNetwork(
                    input_dim=self.global_config.input_dim,
                    output_dim=self.global_config.output_dim,
                    config=self.global_config.model
                ).to(self.device)
                model.load_state_dict(self.model.state_dict())
                
                # Create WorkerOperations for TriSAFE protocol
                if self.worker_loaders and i < len(self.worker_loaders):
                    worker_ops = create_worker(
                        worker_id=i,
                        global_config=self.global_config,
                        model=model,
                        dataset=self.worker_loaders[i].dataset
                    )
                    self.worker_operations[i] = worker_ops
                else:
                    worker_ops = None
                
                # Attack setup if malicious
                attack_trainer = None
                if hasattr(self.global_config, 'malicious_workers') and i in self.global_config.malicious_workers:
                    attack_trainer = AttackTrainer(
                        model=model,
                        train_dataset=self.worker_loaders[i].dataset if self.worker_loaders else None,
                        client_id=i,
                        device=self.device
                    )
                    attack_config = self.global_config.malicious_workers[i]
                    if attack_trainer.configure_attack(attack_config):
                        self.attack_trainers[i] = attack_trainer
                        self.logger.info(f"Configured attack for worker {i}: {attack_config}")
                
                # Create ModelTrainer with WorkerOperations
                trainer = ModelTrainer(model, worker_config, worker_ops=worker_ops)
                if self.worker_loaders is not None and i < len(self.worker_loaders):
                    trainer.train_loader = self.worker_loaders[i]
                
                self.worker_trainers.append(trainer)
                self.workers[i] = trainer
                
            self.logger.info(f"Initialized {len(self.worker_trainers)} workers "
                            f"({len(self.attack_trainers)} malicious) with TriSAFE protocol")
                
        except Exception as e:
            self.logger.error(f"Error initializing workers: {str(e)}")

    def train_round(self, round_num: int) -> Dict[str, Any]:
        """One round of training using TriSAFE protocol"""
        self.logger.info(f"Starting training round {round_num}")
        self.current_round = round_num
        
        try:
            if not hasattr(self, 'workers'):
                self.workers = {}
                self.initialize_workers()

            updates = []  # Will store (gradients, metadata) tuples for server
            
            # Training loop - use WorkerOperations for TriSAFE protocol
            for worker_id, worker in self.workers.items():
                self.logger.info(f"Processing worker {worker_id}")
                
                # Sync model state
                worker.model.load_state_dict(self.model.state_dict())
                
                # Use WorkerOperations if available
                if worker_id in self.worker_operations:
                    worker_ops = self.worker_operations[worker_id]
                    # This returns properly processed gradients with proofs
                    gradients, metadata = worker_ops.train_local_model()
                    
                    # Apply attacks if configured
                    attack_trainer = self.attack_trainers.get(worker_id)
                    if attack_trainer and attack_trainer.is_malicious:
                        # Apply attack to gradients
                        attacked_grads, attack_metrics = attack_trainer.get_malicious_updates(
                            {f'param_{i}': g for i, g in enumerate(gradients)}
                        )
                        gradients = list(attacked_grads.values())
                        
                        # Update metadata with attack info
                        metadata['is_malicious'] = True
                        metadata['attack_type'] = attack_trainer.attack_type
                        metadata['attack_impact'] = attack_metrics.get('attack_impact', 0.0)
                        
                        # Handle time delay attack timestamp manipulation
                        if attack_trainer.attack_type == 'time_delay':
                            proof = attack_trainer.prepare_proof_dictionary(metadata['performance'])
                            metadata['timestamp'] = proof['timestamp']
                    
                    updates.append((gradients, metadata))
                    
                else:
                    # Fallback for workers without WorkerOperations
                    self.logger.warning(f"Worker {worker_id} missing WorkerOperations, using fallback")
                    if hasattr(worker, 'train_loader') and worker.train_loader:
                        metrics = worker.train_with_cross_validation(
                            worker.train_loader, 
                            self.attack_trainers.get(worker_id)
                        )
                        # Extract gradients manually
                        gradients = worker.get_gradients_for_trisafe()
                        metadata = {
                            'worker_id': worker_id,
                            'timestamp': time.time(),
                            'weight': 1.0,
                            'performance': metrics,
                            'bulletproof': {},  # Placeholder
                            'pep_proof': {}  # Placeholder
                        }
                        updates.append((gradients, metadata))
            
            # Process through server with TriSAFE protocol
            if updates:
                success = self.server_ops.process_batch_updates(updates, batch_id=round_num)
                if success:
                    round_metrics = self._compute_round_metrics_from_updates(updates)
                    return round_metrics
            
            return None
                
        except Exception as e:
            self.logger.error(f"Error in training round: {str(e)}")
            return None


    def _compute_round_metrics_from_updates(self, updates: List[Tuple[List[torch.Tensor], Dict]]) -> Dict:
            """Compute round metrics from worker updates"""
            metrics = {
                'round': self.current_round,
                'num_workers': len(updates)
            }
            
            # Aggregate performance metrics
            total_loss = 0.0
            total_acc = 0.0
            malicious_count = 0
            
            for _, metadata in updates:
                if 'performance' in metadata:
                    perf = metadata['performance']
                    total_loss += perf.get('loss', 0.0)
                    total_acc += perf.get('accuracy', 0.0)
                
                if metadata.get('is_malicious', False):
                    malicious_count += 1
            
            metrics['loss'] = total_loss / len(updates) if updates else 0.0
            metrics['accuracy'] = total_acc / len(updates) if updates else 0.0
            metrics['malicious_workers'] = malicious_count
            
            # Add server state
            if hasattr(self.server_ops, 'current_privacy_budget'):
                metrics['privacy_budget_remaining'] = self.server_ops.current_privacy_budget
            
            return metrics

    def _update_client_metrics(self, client_id: int, updates: Dict[str, torch.Tensor],
                             metrics: Dict[str, float]):
        """Track client metrics including validation metrics"""
        try:
            # Ensure client metrics structure exists
            if client_id not in self.client_metrics:
                self.client_metrics[client_id] = {
                    'accuracy_history': [],
                    'loss_history': [],
                    'val_loss_history': [],
                    'val_accuracy_history': [],
                    'learning_rates': []
                }
            
            # Update histories using the correct metric keys
            self.client_metrics[client_id]['accuracy_history'].append(metrics.get('train_accuracy', 0.0))
            self.client_metrics[client_id]['loss_history'].append(metrics.get('train_loss', float('inf')))
            self.client_metrics[client_id]['val_loss_history'].append(metrics.get('val_loss', float('inf')))
            self.client_metrics[client_id]['val_accuracy_history'].append(metrics.get('val_accuracy', 0.0))
            
        except Exception as e:
            self.logger.error(f"Error updating client metrics: {str(e)}")

    def _compute_round_metrics(self, local_metrics: Dict[int, Dict[str, float]]) -> Dict[str, float]:
        """Compute aggregated metrics for the round"""
        try:
            if not local_metrics:
                return {}
                
            avg_metrics = defaultdict(float)
            metrics_count = defaultdict(int)
            
            # Aggregate metrics from all workers
            for worker_metrics in local_metrics.values():
                for k, v in worker_metrics.items():
                    if isinstance(v, (int, float)):
                        avg_metrics[k] += v
                        metrics_count[k] += 1

            # Compute averages and rename keys to match expected format
            averaged_metrics = {}
            for k, v in avg_metrics.items():
                if metrics_count[k] > 0:
                    if k == 'train_loss':
                        averaged_metrics['loss'] = v / metrics_count[k]
                    elif k == 'train_accuracy':
                        averaged_metrics['accuracy'] = v / metrics_count[k]
                    elif k == 'val_loss':
                        averaged_metrics['validation_loss'] = v / metrics_count[k]
                    elif k == 'val_accuracy':
                        averaged_metrics['validation_accuracy'] = v / metrics_count[k]
                    else:
                        averaged_metrics[k] = v / metrics_count[k]

            return averaged_metrics
                    
        except Exception as e:
            self.logger.error(f"Error computing round metrics: {str(e)}")
            return {}

    def _log_attack_status(self, attack_metrics: Dict[int, Dict[str, Any]]):
        """Log comprehensive attack status"""
        self.logger.info("\n=== Round Attack Status and Metrics ===")
        for worker_id, metrics in attack_metrics.items():
            self.logger.info(f"Worker {worker_id} Status:")
            self.logger.info(f"  - Is Malicious: {metrics.get('is_malicious', False)}")
            self.logger.info(f"  - Attack Type: {metrics.get('attack_type', 'None')}")
            self.logger.info(f"  - Attack Effect: {metrics.get('recent_impact', 0.0):.4f}")
            self.logger.info(f"  - Success Rate: {metrics.get('attack_success_rate', 0.0):.4f}")
            self.logger.info(f"  - Updates Modified: {metrics.get('updates_modified', 0)}")
        self.logger.info("="*50)

    def _compute_update_scale(self, metrics: Dict[str, float]) -> float:
        """Compute update scale factor based on validation performance"""
        # Scale updates based on validation loss improvement
        val_loss = metrics.get('val_loss', float('inf'))
        if val_loss < self.best_global_val_loss:
            return 0.5  # Larger updates when validation improves
        return 0.3  # More conservative updates otherwise

    def _aggregate_and_apply_updates(self, updates: Dict[int, Dict[str, torch.Tensor]], 
                                    local_metrics: Dict[int, Dict[str, float]]) -> Dict[str, float]:
        """Legacy aggregation method - kept for compatibility but should use server_ops instead"""
        self.logger.warning("Using legacy aggregation - should use server_ops.process_batch_updates instead")
        
        try:
            # Convert to format expected by server
            batch_updates = []
            for worker_id, update in updates.items():
                gradients = [update[name] for name in sorted(update.keys())]
                metadata = {
                    'worker_id': worker_id,
                    'timestamp': time.time(),
                    'weight': 1.0,
                    'performance': local_metrics.get(worker_id, {}),
                    'bulletproof': {},
                    'pep_proof': {}
                }
                batch_updates.append((gradients, metadata))
            
            # Use server operations
            success = self.server_ops.process_batch_updates(batch_updates, batch_id=self.current_round)
            if success:
                return self._compute_round_metrics(local_metrics)
            return {}
            
        except Exception as e:
            self.logger.error(f"Error in update aggregation: {str(e)}")
            return {}

    def evaluate_global(self, val_loader: DataLoader,
                       test_loader: Optional[DataLoader] = None) -> Dict[str, Dict[str, float]]:
        """Enhanced evaluation with cross-validation results"""
        try:
            worker_config = WorkerConfig(
                global_config=self.global_config,
                worker_id=-1
            )
            eval_trainer = ModelTrainer(self.model, worker_config)
            
            metrics = {'validation': eval_trainer.evaluate(val_loader)}
            
            if test_loader is not None:
                metrics['test'] = eval_trainer.evaluate(test_loader)
                
            # Log evaluation results
            for split, split_metrics in metrics.items():
                self.logger.info(
                    f"{split.capitalize()} metrics - "
                    f"Loss: {split_metrics.get('loss', float('inf')):.4f}, "
                    f"Accuracy: {split_metrics.get('accuracy', 0.0):.4f}"
                )
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in global evaluation: {str(e)}")
            return {
                'validation': {'loss': float('inf'), 'accuracy': 0.0},
                'test': {'loss': float('inf'), 'accuracy': 0.0} if test_loader else None
            }

    def save_checkpoint(self, path: str):
        try:
            checkpoint = {
                'model_state': self.model.state_dict(),
                'round_metrics': dict(self.round_metrics),
                'best_global_val_loss': self.best_global_val_loss,
                'best_global_model_state': self.best_global_model_state,
                'config': {
                    'global': self.global_config.__dict__,
                    'server': self.server_config.__dict__
                },
                'client_metrics': dict(self.client_metrics)
            }
            
            torch.save(checkpoint, path)
            self.logger.info(f"Saved checkpoint to {path}")
            
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {str(e)}")
    
    def load_checkpoint(self, path: str):
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state'])
            self.round_metrics = defaultdict(list, checkpoint['round_metrics'])
            self.best_global_val_loss = checkpoint.get('best_global_val_loss', float('inf'))
            
            if 'best_global_model_state' in checkpoint:
                self.best_global_model_state = checkpoint['best_global_model_state']
            
            self.global_config.__dict__.update(checkpoint['config']['global'])
            self.server_config.__dict__.update(checkpoint['config']['server'])
            
            if 'client_metrics' in checkpoint:
                self.client_metrics = defaultdict(lambda: {
                    'gradient_norms': [],
                    'update_magnitude': [],
                    'accuracy_history': [],
                    'loss_history': [],
                    'val_loss_history': [],
                    'val_accuracy_history': []
                }, checkpoint['client_metrics'])
            
            self.initialize_workers()
            self.logger.info(f"Loaded checkpoint from {path}")
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {str(e)}")
            raise

    def get_model_size(self) -> Dict[str, float]:
        try:
            size_info = {}
            
            total_params = sum(p.numel() for p in self.model.parameters())
            size_info['total_parameters'] = total_params
            
            model_size_bytes = sum(param.numel() * param.element_size() 
                                 for param in self.model.parameters())
            size_info['model_size_mb'] = model_size_bytes / (1024 * 1024)
            
            if torch.cuda.is_available():
                size_info['gpu_memory_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
            
            return size_info
            
        except Exception as e:
            self.logger.error(f"Error getting model size: {str(e)}")
            return {'error': str(e)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='TinyML Federated Training with Cross Validation')
    parser.add_argument('distribution', nargs='?', choices=['iid', 'non-iid'], default='iid',
                       help='Data distribution type (iid or non-iid)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--quantize', action='store_true', help='Enable quantization')
    parser.add_argument('--deploy', action='store_true', help='Prepare for deployment')
    args = parser.parse_args()

    try:
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        global_config = GlobalConfig()
        server_config = ServerConfig(global_config)

        # Create data loaders with specified distribution
        worker_loaders, val_loader, test_loader = create_federated_data_loaders(
            global_config,
            batch_size=args.batch_size,
            distribution_type=args.distribution
        )

        # Initialize trainer
        trainer = FederatedTrainer(
            server_config,
            global_config,
            existing_loaders=(worker_loaders, val_loader, test_loader)
        )

        # Print initial model size
        size_info = trainer.get_model_size()
        trainer.logger.info(f"Initial model size: {size_info}")

        # Training loop with cross-validation
        for epoch in tqdm(range(args.epochs), desc="Training"):
            metrics = trainer.train_round(epoch)
            
            if metrics:
                trainer.logger.info(f"Epoch {epoch} metrics: {metrics}")
            
            # Regular evaluation
            if epoch % global_config.evaluation_frequency == 0:
                eval_metrics = trainer.evaluate_global(val_loader, test_loader)
                trainer.logger.info(f"Evaluation metrics: {eval_metrics}")
                
            # Save checkpoints
            if epoch % global_config.checkpoint_frequency == 0:
                trainer.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")

        # Load best model for final evaluation
        if trainer.best_global_model_state is not None:
            trainer.model.load_state_dict(trainer.best_global_model_state)
            final_metrics = trainer.evaluate_global(val_loader, test_loader)
            trainer.logger.info(f"Final metrics with best model: {final_metrics}")

        # Post-training processing
        if args.quantize or args.deploy:
            # Quantization/deployment logic would go here
            final_size_info = trainer.get_model_size()
            trainer.logger.info(f"Final model size after optimization: {final_size_info}")
            trainer.save_checkpoint("optimized_model.pt")
                
    except Exception as e:
        logging.error(f"Training error: {str(e)}", exc_info=True)
        raise