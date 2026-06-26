"""TriSAFE Federated Learning System"""
import os
import sys
import time
import json
import random
import math  # Add this
import numpy as np
import torch
import torch.nn.functional as F  # Add this
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
from collections import defaultdict

# Import TriSAFE components
from config import (
    GlobalConfig, 
    create_default_config, 
    create_test_config,
    create_production_config,
    save_config_to_file,
    load_config_from_file
)
from server_ops import ServerOperations, ServerConfig
from worker_ops import WorkerOperations, create_worker
from data_loader import create_federated_data_loaders


class TriSAFESystem:
    """Main TriSAFE Federated Learning System"""
    
    def __init__(self, config: GlobalConfig, args: argparse.Namespace):
        """Initialize TriSAFE system with given configuration"""
        self.config = config
        self.args = args
        self.logger = self._setup_logging()
        
        # Create experiment directory
        self.experiment_dir = self._create_experiment_dir()
        
        # First, load data to determine actual dimensions
        self.worker_loaders, self.val_loader, self.test_loader = create_federated_data_loaders(
            self.config,
            distribution_type=args.distribution
        )
        
        # Determine actual data dimensions from loaded data
        self._detect_data_dimensions()
        
        # Initialize model with correct dimensions
        self.model = self._create_model()
        
        # Initialize server
        self.server = self._initialize_server()
        
        # Initialize workers with correct model architecture
        self.workers = self._initialize_workers()
        
        # Metrics tracking
        self.metrics_history = []
        self.best_accuracy = 0.0
        self.current_round = 0
        
        self.logger.info(f"TriSAFE System initialized with {len(self.workers)} workers")
        self.logger.info(f"Data dimensions: input={self.input_dim}, output={self.output_dim}")
        self.logger.info(f"Privacy budget: ε={config.privacy_budget}, δ={config.delta}")
        self.logger.info(f"Threshold scheme: {config.threshold_t}-of-{config.threshold_n}")
        self.logger.info(f"Data distribution: {args.distribution}")
    
    def _detect_data_dimensions(self):
        """Detect actual input and output dimensions from loaded data"""
        # Get a sample batch to determine dimensions
        sample_loader = self.worker_loaders[0] if self.worker_loaders else self.val_loader
        
        for data, labels in sample_loader:
            # Handle different data formats
            if len(data.shape) > 2:
                # Flatten if needed (e.g., images)
                batch_size = data.shape[0]
                self.input_dim = data.view(batch_size, -1).shape[1]
            else:
                self.input_dim = data.shape[1]
            
            # Determine number of classes
            if len(labels.shape) > 1 and labels.shape[1] > 1:
                # One-hot encoded
                self.output_dim = labels.shape[1]
            else:
                # Class indices
                self.output_dim = len(torch.unique(labels))
            
            self.logger.info(f"Detected data dimensions: input_dim={self.input_dim}, output_dim={self.output_dim}")
            break
        
        # Update config with detected dimensions
        self.config.input_dim = self.input_dim
        self.config.output_dim = self.output_dim
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Create logs directory
        Path('logs').mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'logs/trisafe_{datetime.now():%Y%m%d_%H%M%S}.log')
            ]
        )
        
        return logging.getLogger('TriSAFE')
    
    def _create_experiment_dir(self) -> Path:
        """Create directory for experiment results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{self.args.experiment_name}_{timestamp}"
        experiment_dir = Path("experiments") / exp_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_path = experiment_dir / "config.json"
        save_config_to_file(self.config, str(config_path))
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'experiment_name': self.args.experiment_name,
            'num_workers': self.config.num_workers,
            'num_rounds': self.config.num_rounds,
            'privacy_budget': self.config.privacy_budget,
            'byzantine_threshold': self.config.byzantine_threshold,
            'data_distribution': self.args.distribution,
            'dataset': self.args.dataset,
            'detected_input_dim': getattr(self, 'input_dim', 'unknown'),
            'detected_output_dim': getattr(self, 'output_dim', 'unknown'),
            'attack_config': {
                'type': self.args.attack_type,
                'fraction': self.args.attack_fraction
            } if self.args.attack_type else None,
            'device': self.config.device
        }
        
        with open(experiment_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)
        
        return experiment_dir
    
    def _create_model(self) -> torch.nn.Module:
        """Create the neural network model based on detected dimensions"""
        import torch.nn as nn
        
        # For Edge-IIoT dataset (61 features, typically binary/multi-class classification)
        if self.input_dim == 61:
            self.logger.info("Creating model for Edge-IIoT dataset")
            model = nn.Sequential(
                nn.Linear(self.input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, self.output_dim)
            )
        # For MNIST-like data
        elif self.input_dim == 784:
            self.logger.info("Creating model for MNIST-like dataset")
            model = nn.Sequential(
                nn.Linear(784, self.config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.config.hidden_dim, self.output_dim)
            )
        # For CIFAR-like data
        elif self.input_dim == 3072:  # 32x32x3
            self.logger.info("Creating model for CIFAR-like dataset")
            model = nn.Sequential(
                nn.Linear(3072, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, self.output_dim)
            )
        else:
            # Generic model for other dimensions
            self.logger.info(f"Creating generic model for input_dim={self.input_dim}")
            hidden_size = min(max(self.input_dim * 2, 64), 512)  # Adaptive hidden size
            model = nn.Sequential(
                nn.Linear(self.input_dim, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size // 2, self.output_dim)
            )
        
        return model.to(torch.device(self.config.device))
    
    def _initialize_server(self) -> ServerOperations:
        """Initialize server with TriSAFE protocol"""
        server_config = ServerConfig(
            global_config=self.config,
            num_workers=self.config.num_workers,
            batch_size=self.config.train_batch_size,
            learning_rate=self.config.learning_rate,
            momentum=self.config.momentum,
            byzantine_threshold=self.config.byzantine_threshold,
            time_window=self.config.time_window,
            cover_traffic_ratio=self.config.cover_traffic_ratio
        )
        
        return ServerOperations(server_config, self.model)
    
    def _initialize_workers(self) -> List[WorkerOperations]:
        """Initialize all workers with correct model"""
        workers = []
        
        for worker_id in range(self.config.num_workers):
            # Create worker with same model architecture
            worker_model = self._create_model()
            
            worker = create_worker(
                worker_id=worker_id,
                global_config=self.config,
                model=worker_model,
                dataset=self.worker_loaders[worker_id].dataset
            )
            workers.append(worker)
        
        # Configure attacks if specified
        if self.args.attack_type:
            self._configure_attacks(workers)
        
        return workers
    
    def _configure_attacks(self, workers: List[WorkerOperations]):
        """Configure Byzantine attacks for specified fraction of workers"""
        num_byzantine = int(self.config.num_workers * self.args.attack_fraction)
        if num_byzantine == 0:
            return
        
        # Randomly select Byzantine workers
        byzantine_ids = random.sample(range(self.config.num_workers), num_byzantine)
        
        for worker_id in byzantine_ids:
            worker = workers[worker_id]
            # Mark worker as Byzantine
            worker.is_byzantine = True
            worker.attack_type = self.args.attack_type
            
        self.logger.info(f"Configured {num_byzantine} Byzantine workers: {byzantine_ids}")
        self.logger.info(f"Attack type: {self.args.attack_type}")

    def _apply_attack(self, worker, worker_idx, gradients):
        """Apply the configured Byzantine attack to a malicious worker's gradients.

        Returns (possibly-modified gradients, submission timestamp). Honest workers
        are returned unchanged. Implemented attacks are dataset-agnostic and operate
        in gradient / timing space:
          - 'sign_flip'  : negate every gradient coordinate
          - 'byzantine'  : strong directed corruption (scaled negation); the server's
                           VRS clipping/weight-caps bound its accepted influence
          - 'noise'      : add large Gaussian noise to each coordinate
          - 'time_delay' : leave gradients intact but stamp a late arrival so that
                           Layer-1 fixed-window enforcement drops the update
        """
        timestamp = time.time()
        if not getattr(worker, 'is_byzantine', False):
            return gradients, timestamp

        attack_type = getattr(worker, 'attack_type', None)

        if attack_type == 'sign_flip':
            gradients = [(-g if g is not None else None) for g in gradients]

        elif attack_type == 'byzantine':
            scale = 5.0
            gradients = [(-scale * g if g is not None else None) for g in gradients]

        elif attack_type == 'noise':
            noisy = []
            for g in gradients:
                if g is None:
                    noisy.append(None)
                else:
                    magnitude = g.abs().mean().item() + 1e-6
                    noisy.append(g + torch.randn_like(g) * magnitude * 10.0)
            gradients = noisy

        elif attack_type == 'time_delay':
            # Submit after the window so Layer-1 marks the update as a late arrival.
            timestamp = time.time() + float(self.config.time_window) + 1.0

        else:
            self.logger.warning(
                f"Worker {worker_idx}: unknown attack_type '{attack_type}', behaving honestly"
            )

        self.logger.debug(f"Worker {worker_idx}: applied attack '{attack_type}'")
        return gradients, timestamp
    
    def train(self):
        """Main training loop implementing TriSAFE protocol"""
        self.logger.info("Starting TriSAFE training...")
        self.logger.info(f"Configuration: {self.config.num_rounds} rounds, {self.config.num_workers} workers")
        self.logger.info(f"Model architecture: {self.model}")
        
        for round_idx in tqdm(range(self.config.num_rounds), desc="Training"):
            self.current_round = round_idx
            round_start = time.time()
            
            try:
                # Execute training round
                round_metrics = self._train_round(round_idx)
                
                if round_metrics:
                    # Add timing
                    round_metrics['duration'] = time.time() - round_start
                    self.metrics_history.append(round_metrics)
                    
                    # Log progress
                    self._log_progress(round_idx, round_metrics)
                    
                    # Save checkpoint if needed
                    if round_idx % self.config.checkpoint_frequency == 0:
                        self._save_checkpoint(round_idx)
                    
                    # Check for early stopping
                    if self._should_stop_early(round_metrics):
                        self.logger.info("Early stopping triggered")
                        break
                
            except Exception as e:
                self.logger.error(f"Error in round {round_idx}: {str(e)}", exc_info=True)
                continue
        
        # Save final results
        self._save_final_results()
        self.logger.info("Training completed")
    


    def _train_round(self, round_idx: int) -> Dict[str, Any]:
        """
        Execute one training round following TriSAFE protocol
        Ensures float gradients are passed to server for proper processing
        """
        round_metrics = {'round': round_idx}
        
        try:
            # Step 1: Sync all workers with current server model
            server_state_dict = self.model.state_dict()
            for worker in self.workers:
                worker.model.load_state_dict(server_state_dict)
            
            # Step 2: Collect worker updates (float gradients)
            worker_updates = []
            
            for worker_idx, worker in enumerate(self.workers):
                try:
                    # Compute gradients through local training
                    gradients = None
                    accuracy = 0.0
                    loss = float('inf')
                    
                    # Check if we have a data loader for this worker
                    if self.worker_loaders and worker_idx < len(self.worker_loaders):
                        data_loader = self.worker_loaders[worker_idx]
                        
                        # Perform one epoch of training to compute gradients
                        worker.model.train()
                        optimizer = torch.optim.SGD(worker.model.parameters(), lr=self.config.learning_rate)
                        
                        total_loss = 0.0
                        correct = 0
                        total = 0
                        accumulated_grads = None
                        batch_count = 0
                        
                        # Train for the configured number of local epochs (full passes
                        # over the worker's data). Previously this loop broke after
                        # `local_epochs - 1` BATCHES, i.e. a single batch when
                        # local_epochs=1 -- a misuse of the epoch parameter.
                        num_local_epochs = max(1, int(self.config.local_epochs))
                        for _local_epoch in range(num_local_epochs):
                            for batch_idx, (data, target) in enumerate(data_loader):
                                # Move to device
                                data = data.to(self.config.device)
                                target = target.to(self.config.device)

                                # Ensure proper shape
                                if len(data.shape) > 2:
                                    data = data.view(data.size(0), -1)

                                # Forward pass
                                optimizer.zero_grad()
                                output = worker.model(data)
                                batch_loss = F.cross_entropy(output, target)

                                # Backward pass to compute gradients
                                batch_loss.backward()

                                # Accumulate gradients (don't apply them yet)
                                if accumulated_grads is None:
                                    accumulated_grads = [p.grad.clone().detach() for p in worker.model.parameters()]
                                else:
                                    for i, p in enumerate(worker.model.parameters()):
                                        if p.grad is not None:
                                            accumulated_grads[i] += p.grad.clone().detach()

                                # Track metrics
                                total_loss += batch_loss.item() * data.size(0)
                                pred = output.argmax(dim=1)
                                correct += pred.eq(target).sum().item()
                                total += target.size(0)
                                batch_count += 1
                        
                        # Average the accumulated gradients
                        if accumulated_grads and batch_count > 0:
                            gradients = [g / batch_count for g in accumulated_grads]
                            accuracy = correct / total if total > 0 else 0.0
                            loss = total_loss / total if total > 0 else float('inf')
                        else:
                            gradients = [torch.zeros_like(p) for p in worker.model.parameters()]
                    
                    else:
                        # No data loader available, use zero gradients
                        gradients = [torch.zeros_like(p) for p in worker.model.parameters()]
                    
                    # Step 3: Apply gradient clipping (L2 norm)
                    total_norm = 0.0
                    for grad in gradients:
                        if grad is not None:
                            total_norm += torch.norm(grad).item() ** 2
                    total_norm = math.sqrt(total_norm)
                    
                    if total_norm > self.config.max_grad_norm:
                        clip_factor = self.config.max_grad_norm / total_norm
                        gradients = [g * clip_factor if g is not None else None for g in gradients]
                        self.logger.debug(f"Worker {worker_idx}: Clipped gradients from {total_norm:.4f} to {self.config.max_grad_norm}")
                    
                    # Step 4: Ensure gradients are float tensors (not int)
                    float_gradients = []
                    for grad in gradients:
                        if grad is not None:
                            if grad.dtype in [torch.int32, torch.int64]:
                                # Convert back to float if accidentally converted to int
                                float_gradients.append(grad.float())
                            else:
                                float_gradients.append(grad)
                        else:
                            float_gradients.append(None)
                    
                    # Apply the configured Byzantine attack (if this worker is
                    # malicious). Previously workers were flagged byzantine but the
                    # attack was never applied, so they behaved honestly.
                    float_gradients, attack_timestamp = self._apply_attack(
                        worker, worker_idx, float_gradients
                    )

                    # Step 5: Create metadata for this worker
                    metadata = {
                        'worker_id': worker_idx,
                        'timestamp': attack_timestamp,
                        'weight': 1.0,  # Raw weight w_i
                        'performance': {
                            'accuracy': accuracy,
                            'loss': loss
                        },
                        'gradient_norm': total_norm,
                        # Placeholder proofs - in production, use WorkerOperations to generate real proofs
                        'bulletproof': {},
                        'pep_proof': {}
                    }
                    
                    # Add to batch
                    worker_updates.append((float_gradients, metadata))
                    
                    self.logger.debug(f"Worker {worker_idx}: loss={loss:.4f}, acc={accuracy:.4f}, norm={total_norm:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing worker {worker_idx}: {str(e)}")
                    # Add zero gradients for failed worker
                    zero_grads = [torch.zeros_like(p) for p in self.model.parameters()]
                    metadata = {
                        'worker_id': worker_idx,
                        'timestamp': time.time(),
                        'weight': 0.0,  # Zero weight for failed worker
                        'performance': {'accuracy': 0.0, 'loss': float('inf')},
                        'bulletproof': {},
                        'pep_proof': {}
                    }
                    worker_updates.append((zero_grads, metadata))
            
            # Step 6: Process through TriSAFE server protocol
            success = self.server.process_batch_updates(worker_updates, round_idx)
            
            if not success:
                self.logger.warning(f"Round {round_idx} processing failed")
                return None
            
            # Step 7: Evaluate model if needed
            if round_idx % self.config.validation_frequency == 0:
                val_metrics = self._evaluate('validation')
                round_metrics.update(val_metrics)
                
                # Update best model if improved
                if val_metrics.get('val_accuracy', 0) > self.best_accuracy:
                    self.best_accuracy = val_metrics['val_accuracy']
                    self._save_best_model()
            
            # Step 8: Update privacy and round metrics
            if hasattr(self.server, 'current_privacy_budget'):
                round_metrics['privacy_budget_remaining'] = self.server.current_privacy_budget
            round_metrics['iteration'] = round_idx
            round_metrics['num_workers'] = len(worker_updates)
            
            # Add aggregated performance metrics
            total_accuracy = sum(m['performance']['accuracy'] for _, m in worker_updates) / len(worker_updates)
            avg_loss = sum(m['performance']['loss'] for _, m in worker_updates if m['performance']['loss'] < float('inf')) 
            num_valid = sum(1 for _, m in worker_updates if m['performance']['loss'] < float('inf'))
            avg_loss = avg_loss / num_valid if num_valid > 0 else float('inf')
            
            round_metrics['train_accuracy'] = total_accuracy
            round_metrics['train_loss'] = avg_loss
            
            self.logger.info(f"Round {round_idx} complete: acc={total_accuracy:.4f}, loss={avg_loss:.4f}")
            
            return round_metrics
            
        except Exception as e:
            self.logger.error(f"Error in round {round_idx}: {str(e)}", exc_info=True)
            return None



    
    def _evaluate(self, mode: str = 'validation') -> Dict[str, float]:
        """Evaluate model performance"""
        import torch.nn.functional as F
        
        loader = self.val_loader if mode == 'validation' else self.test_loader
        
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in loader:
                data = data.to(self.config.device)
                target = target.to(self.config.device)
                
                # Ensure data has correct shape
                if len(data.shape) > 2:
                    data = data.view(data.size(0), -1)
                
                output = self.model(data)
                
                # Handle different label formats
                if len(target.shape) > 1 and target.shape[1] > 1:
                    # One-hot encoded labels
                    loss = F.cross_entropy(output, target.argmax(dim=1))
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.argmax(dim=1).view_as(pred)).sum().item()
                else:
                    # Class indices
                    loss = F.cross_entropy(output, target.long())
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.long().view_as(pred)).sum().item()
                
                total_loss += loss.item() * data.size(0)
                total += target.size(0)
        
        accuracy = correct / total if total > 0 else 0
        avg_loss = total_loss / total if total > 0 else 0
        
        return {
            f'{mode[:3]}_accuracy': accuracy,
            f'{mode[:3]}_loss': avg_loss
        }
    
    def _log_progress(self, round_idx: int, metrics: Dict):
        """Log training progress"""
        log_items = []
        for key, value in metrics.items():
            if isinstance(value, float):
                log_items.append(f"{key}={value:.4f}")
        
        self.logger.info(f"Round {round_idx}: {' '.join(log_items)}")
    
    def _save_checkpoint(self, round_idx: int):
        """Save training checkpoint"""
        checkpoint = {
            'round': round_idx,
            'model_state_dict': self.model.state_dict(),
            'server_state': self.server.get_model_state(),
            'metrics_history': self.metrics_history,
            'best_accuracy': self.best_accuracy,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'config': self.config.__dict__
        }
        
        checkpoint_path = self.experiment_dir / f"checkpoint_round_{round_idx}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_best_model(self):
        """Save best performing model"""
        model_path = self.experiment_dir / "best_model.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'accuracy': self.best_accuracy,
            'round': self.current_round,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim
        }, model_path)
        self.logger.info(f"Best model saved with accuracy {self.best_accuracy:.4f}")
    
    def _should_stop_early(self, metrics: Dict) -> bool:
        """Check early stopping condition"""
        if not hasattr(self, 'patience_counter'):
            self.patience_counter = 0
        
        if 'val_accuracy' in metrics:
            if metrics['val_accuracy'] <= self.best_accuracy:
                self.patience_counter += 1
            else:
                self.patience_counter = 0
        
        return self.patience_counter >= 10
    
    def _save_final_results(self):
        """Save final training results"""
        # Final evaluation
        test_metrics = self._evaluate('test')
        
        results = {
            'final_round': self.current_round,
            'best_validation_accuracy': self.best_accuracy,
            'test_metrics': test_metrics,
            'privacy_budget_used': self.config.privacy_budget - self.server.current_privacy_budget,
            'total_rounds': len(self.metrics_history),
            'data_dimensions': {
                'input_dim': self.input_dim,
                'output_dim': self.output_dim
            },
            'configuration': {
                'num_workers': self.config.num_workers,
                'num_rounds': self.config.num_rounds,
                'privacy_budget': self.config.privacy_budget,
                'noise_multiplier': self.config.noise_multiplier,
                'byzantine_threshold': self.config.byzantine_threshold,
                'data_distribution': self.args.distribution,
                'dataset': self.args.dataset
            },
            'attack_config': {
                'type': self.args.attack_type,
                'fraction': self.args.attack_fraction,
                'num_byzantine': int(self.config.num_workers * self.args.attack_fraction)
            } if self.args.attack_type else None,
            'metrics_history': self.metrics_history
        }
        
        results_path = self.experiment_dir / "final_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        self.logger.info(f"Final results saved to {results_path}")
        self.logger.info(f"Test accuracy: {test_metrics.get('tes_accuracy', 0):.4f}")
        self.logger.info(f"Best validation accuracy: {self.best_accuracy:.4f}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='TriSAFE Federated Learning System')
    
    # Basic arguments
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--mode', choices=['default', 'test', 'production'], 
                       default='default', help='Configuration mode')
    parser.add_argument('--experiment_name', type=str, default='trisafe',
                       help='Name for this experiment')
    
    # Data arguments
    parser.add_argument('--dataset', choices=['edge_iiot', 'nbaiot', 'mnist', 'cifar10', 'custom'],
                       default='edge_iiot', help='Dataset to use')
    parser.add_argument('--distribution', choices=['iid', 'non_iid'],
                       default='iid', help='Data distribution type')
    
    # Training arguments
    parser.add_argument('--num_rounds', type=int, help='Number of training rounds')
    parser.add_argument('--num_workers', type=int, help='Number of workers')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    
    # Privacy arguments
    parser.add_argument('--privacy_budget', type=float, help='Privacy budget (epsilon)')
    parser.add_argument('--noise_multiplier', type=float, help='Noise multiplier')
    
    # Attack arguments
    parser.add_argument('--attack_type',
                       choices=['sign_flip', 'noise', 'byzantine', 'time_delay'],
                       help='Type of Byzantine attack (gradient/timing-space attacks '
                            'applied in main.py; the full HIDRA/label-flip/FANG suite '
                            'runs via the AttackTrainer pipeline)')
    parser.add_argument('--attack_fraction', type=float, default=0.0,
                       help='Fraction of Byzantine workers (0.0 to 1.0)')
    
    # System arguments
    parser.add_argument('--device', choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING'],
                       default='INFO', help='Logging level')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Load or create configuration
    if args.config:
        config = load_config_from_file(args.config)
    elif args.mode == 'test':
        config = create_test_config()
    elif args.mode == 'production':
        config = create_production_config()
    else:
        config = create_default_config()
    
    # Override config with command line arguments
    if args.num_rounds:
        config.num_rounds = args.num_rounds
    if args.num_workers:
        config.num_workers = args.num_workers
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.privacy_budget:
        config.privacy_budget = args.privacy_budget
    if args.noise_multiplier:
        config.noise_multiplier = args.noise_multiplier
    if args.device:
        config.device = args.device
    if args.log_level:
        config.log_level = args.log_level
    
    # Update dataset in config
    config.dataset = args.dataset
    
    # Validate configuration
    if not config.validate_for_production() and args.mode == 'production':
        print("Warning: Configuration may not be suitable for production")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Initialize and run system
    print(f"Initializing TriSAFE System...")
    print(f"Dataset: {args.dataset}")
    print(f"Distribution: {args.distribution}")
    print(f"Workers: {config.num_workers}")
    print(f"Rounds: {config.num_rounds}")
    
    if args.attack_type:
        print(f"Attack: {args.attack_type} with {args.attack_fraction*100:.1f}% Byzantine workers")
    
    system = TriSAFESystem(config, args)
    
    try:
        system.train()
        print(f"\nTraining completed successfully!")
        print(f"Results saved to: {system.experiment_dir}")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()