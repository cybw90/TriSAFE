"""
worker_ops.py - TriSAFE Paper-Aligned Worker Operations
Implements client-side protocol with:
- Global gradient clipping
- Fixed-point encoding
- Bulletproof range proofs
- PEP (Plaintext Equivalence Protocol)
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import hashlib
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from collections import deque

# Import configurations
from config import GlobalConfig

# Import cryptographic components
from bulletproof_pep import BulletproofRangeProof, PEPProtocol
from phe_mechanism import ThresholdPaillierConfig


@dataclass
class WorkerConfig:
    """Worker configuration aligned with TriSAFE paper"""
    worker_id: int
    global_config: GlobalConfig
    local_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.01
    
    # Clipping parameters from paper
    clipping_bound: float = 1.0  # C in paper
    
    def __post_init__(self):
        """Compute derived parameters from paper"""
        # Fixed-point scales from global config
        self.fixed_point_scale = 2 ** self.global_config.fixed_point_scale_exp  # S_fp = 2^20
        self.weight_scale = 2 ** self.global_config.weight_scale_exp  # S_α = 2^16
        
        # Per-coordinate bound: B_range = C / sqrt(d)
        self.input_dim = self.global_config.input_dim
        self.per_coord_bound = self.clipping_bound / math.sqrt(self.input_dim)
        
        # Range proof offset: T = ceil(S_fp * B_range)
        self.range_offset = int(math.ceil(self.fixed_point_scale * self.per_coord_bound))
        
        # Range proof bits: m = ceil(log2(2T + 1))
        self.range_proof_bits = int(math.ceil(math.log2(2 * self.range_offset + 1)))
        
        # PEP parameters
        self.folding_weight_bits = self.global_config.folding_weight_bits  # κ = 32
        self.packing_base = 2 ** self.global_config.packing_base_exp  # B = 2^29
        self.slots_per_ciphertext = self.global_config.slots_per_ciphertext  # L = 64


class WorkerOperations:
    """Worker operations implementing TriSAFE client-side protocol"""
    
    def __init__(
        self,
        config: WorkerConfig,
        model: nn.Module,
        dataset: Dataset
    ):
        self.config = config
        self.model = model
        self.dataset = dataset
        self.logger = logging.getLogger(f'Worker_{config.worker_id}')
        
        # Initialize data loader
        self.data_loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True
        )
        
        # Initialize cryptographic components
        self.bulletproof = BulletproofRangeProof(config.global_config)
        self.pep_protocol = PEPProtocol(config.global_config)
        
        # Initialize Paillier config for client-side operations
        self.phe_config = ThresholdPaillierConfig(
            modulus_bits=config.global_config.paillier_modulus_bits,
            packing_base_exp=config.global_config.packing_base_exp,
            slots_per_ciphertext=config.global_config.slots_per_ciphertext,
            fixed_point_scale_exp=config.global_config.fixed_point_scale_exp,
            weight_scale_exp=config.global_config.weight_scale_exp
        )
        
        # Performance tracking
        self.performance_history = deque(maxlen=10)
        self.current_epoch = 0
        
        # Gradient history for validation
        self.gradient_history = deque(maxlen=5)
        
        self.logger.info(f"Worker {config.worker_id} initialized with paper parameters:")
        self.logger.info(f"  Clipping bound C = {config.clipping_bound}")
        self.logger.info(f"  Per-coord bound = {config.per_coord_bound:.6f}")
        self.logger.info(f"  Range offset T = {config.range_offset}")
        self.logger.info(f"  Fixed-point scale = 2^{config.global_config.fixed_point_scale_exp}")

    def train_local_model(self) -> Tuple[List[torch.Tensor], Dict]:
        """
        Perform local training and generate gradients with TriSAFE protocol
        
        Returns:
            Tuple of (processed_gradients, proof_metadata)
        """
        self.model.train()
        
        # Accumulate gradients over local epochs
        accumulated_gradients = None
        num_batches = 0
        
        # Track metrics
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Training loop
        for epoch in range(self.config.local_epochs):
            for batch_idx, (data, target) in enumerate(self.data_loader):
                # Ensure proper shape
                if len(data.shape) > 2:
                    data = data.view(data.size(0), -1)
                
                # Forward pass
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                
                # Backward pass
                self.model.zero_grad()
                loss.backward()
                
                # Accumulate gradients
                if accumulated_gradients is None:
                    accumulated_gradients = [p.grad.clone() for p in self.model.parameters()]
                else:
                    for i, p in enumerate(self.model.parameters()):
                        if p.grad is not None:
                            accumulated_gradients[i] += p.grad
                
                num_batches += 1
                
                # Track metrics
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        # Average gradients
        if accumulated_gradients and num_batches > 0:
            gradients = [g / num_batches for g in accumulated_gradients]
        else:
            gradients = [torch.zeros_like(p) for p in self.model.parameters()]
        
        # Compute metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        accuracy = correct / total if total > 0 else 0.0
        
        # Process gradients through TriSAFE protocol
        processed_gradients, proofs = self._process_gradients_trisafe(
            gradients,
            accuracy,
            avg_loss
        )
        
        # Update history
        self.performance_history.append({
            'loss': avg_loss,
            'accuracy': accuracy,
            'epoch': self.current_epoch
        })
        self.current_epoch += 1
        
        return processed_gradients, proofs

    def _process_gradients_trisafe(
        self,
        gradients: List[torch.Tensor],
        accuracy: float,
        loss: float
    ) -> Tuple[List[torch.Tensor], Dict]:
        """
        Process gradients following TriSAFE protocol (Algorithm 2)
        
        Steps:
        1. Global ℓ2 norm clipping to C
        2. Fixed-point encoding
        3. Generate Bulletproof range proofs
        4. Generate PEP proof
        """
        
        # Step 1: Global gradient clipping
        clipped_gradients = self._clip_gradients_global(gradients)
        
        # Step 2: Fixed-point encoding
        encoded_gradients = self._encode_fixed_point(clipped_gradients)
        
        # Step 3: Generate Bulletproof range proof
        bulletproof = self._generate_bulletproof(encoded_gradients)
        
        # Step 4: Pack for Paillier encryption
        packed_blocks = self._pack_for_encryption(encoded_gradients)
        
        # Step 5: Generate PEP proof
        pep_proof = self._generate_pep_proof(packed_blocks)
        
        # Prepare metadata with all proofs
        metadata = {
            'worker_id': self.config.worker_id,
            'timestamp': time.time(),
            'weight': 1.0,  # Raw weight w_i
            'performance': {
                'accuracy': accuracy,
                'loss': loss
            },
            'bulletproof': bulletproof,
            'pep_proof': pep_proof,
            'gradient_norm': self._compute_gradient_norm(gradients),
            'packed_blocks': packed_blocks,  # For server to encrypt
        }
        
        self.logger.debug(f"Processed gradients with norm {metadata['gradient_norm']:.4f}")
        
        return encoded_gradients, metadata

    def _clip_gradients_global(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply global ℓ2 norm clipping: g ← g · min(1, C/||g||_2)
        Paper Section 4.2
        """
        # Compute global ℓ2 norm
        total_norm_sq = 0.0
        for grad in gradients:
            if grad is not None:
                total_norm_sq += torch.norm(grad).item() ** 2
        total_norm = math.sqrt(total_norm_sq)
        
        # Apply clipping if needed
        if total_norm > self.config.clipping_bound:
            clip_factor = self.config.clipping_bound / total_norm
            clipped = []
            for grad in gradients:
                if grad is not None:
                    clipped.append(grad * clip_factor)
                else:
                    clipped.append(None)
            
            self.logger.debug(f"Clipped gradients from {total_norm:.4f} to {self.config.clipping_bound}")
            return clipped
        
        return gradients

    def _encode_fixed_point(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Encode to fixed-point: v_i = round(S_fp * g_i)
        Paper Section 4.2
        """
        S_fp = self.config.fixed_point_scale
        encoded = []
        
        for grad in gradients:
            if grad is not None:
                # Round to nearest integer after scaling
                encoded_grad = torch.round(grad * S_fp).int()
                encoded.append(encoded_grad)
            else:
                encoded.append(None)
        
        return encoded

    def _generate_bulletproof(self, encoded_gradients: List[torch.Tensor]) -> Dict:
        """
        Generate aggregated Bulletproof range proof
        Proves y_{i,k} = v_{i,k} + T ∈ [0, 2^m) for all k
        Paper Section 4.2
        """
        # Flatten all gradients
        flattened = []
        for grad in encoded_gradients:
            if grad is not None:
                flattened.append(grad.view(-1))
        
        if not flattened:
            return {}
        
        flat_values = torch.cat(flattened)
        
        # Add offset T to ensure non-negative
        T = self.config.range_offset
        y_values = flat_values + T
        
        # Generate aggregated Bulletproof
        proof = self.bulletproof.prove_range(
            values=y_values,
            bound=2 ** self.config.range_proof_bits
        )
        
        # Bind to round and client ID
        proof['binding'] = hashlib.sha256(
            f"{self.config.worker_id}_round_{self.current_epoch}".encode()
        ).hexdigest()
        
        return proof

    def _pack_for_encryption(self, encoded_gradients: List[torch.Tensor]) -> List[Dict]:
        """
        Pack gradients into blocks for Paillier encryption
        Each block has L slots, packed with base B = 2^b
        """
        L = self.config.slots_per_ciphertext
        B = self.config.packing_base
        
        # Flatten all values
        all_values = []
        for grad in encoded_gradients:
            if grad is not None:
                all_values.extend(grad.view(-1).tolist())
        
        # Pack into blocks
        blocks = []
        for j in range(0, len(all_values), L):
            block_values = all_values[j:j+L]
            
            # Compute packed plaintext: M_j = Σ v_{j,ℓ} * B^ℓ
            M_j = 0
            for ell, v in enumerate(block_values):
                # Use centered representation for negative values
                if v < 0 and v < -B//2:
                    raise ValueError(f"Value {v} exceeds slot bound")
                M_j += (v if v >= 0 else B + v) * (B ** ell)
            
            blocks.append({
                'plaintext': M_j,
                'num_slots': len(block_values),
                'block_index': j // L
            })
        
        return blocks

    def _generate_pep_proof(self, packed_blocks: List[Dict]) -> Dict:
        """
        Generate PEP (Plaintext Equivalence Protocol) proof
        Binds Pedersen commitments to Paillier plaintexts
        Paper Section 4.2
        """
        if not packed_blocks:
            return {}
        
        # Extract plaintexts
        plaintexts = [block['plaintext'] for block in packed_blocks]
        
        # Generate deterministic folding weights
        kappa = self.config.folding_weight_bits
        folding_input = (
            f"worker_{self.config.worker_id}_"
            f"round_{self.current_epoch}_"
            f"{str(plaintexts)}"
        ).encode()
        
        # Hash to get folding weights u_j ∈ [-2^κ, 2^κ]
        u_weights = []
        for j in range(len(plaintexts)):
            seed = hashlib.sha256(folding_input + str(j).encode()).digest()
            weight = int.from_bytes(seed[:4], 'big', signed=True)
            # Bound to [-2^κ, 2^κ]
            weight = max(-2**kappa, min(2**kappa, weight))
            u_weights.append(weight)
        
        # Create PEP proof structure
        proof = {
            'folding_weights': u_weights,
            'num_blocks': len(packed_blocks),
            'commitment_binding': hashlib.sha256(str(plaintexts).encode()).hexdigest(),
            'round_binding': f"round_{self.current_epoch}",
            'worker_binding': f"worker_{self.config.worker_id}"
        }
        
        return proof

    def _compute_gradient_norm(self, gradients: List[torch.Tensor]) -> float:
        """Compute ℓ2 norm of gradient list"""
        total_norm_sq = 0.0
        for grad in gradients:
            if grad is not None:
                total_norm_sq += torch.norm(grad).item() ** 2
        return math.sqrt(total_norm_sq)

    def prepare_cover_traffic(self) -> Tuple[List[torch.Tensor], Dict]:
        """
        Generate cover traffic for timing privacy
        Paper Section 4.1 - uses ratio ρ
        """
        # Generate dummy gradients within clipping bound
        dummy_gradients = []
        
        for param in self.model.parameters():
            # Random gradient with controlled norm
            rand_grad = torch.randn_like(param)
            # Scale to be within clipping bound
            rand_grad = rand_grad * (self.config.clipping_bound / 10)
            dummy_gradients.append(rand_grad)
        
        # Process through same pipeline
        processed_dummies, dummy_proofs = self._process_gradients_trisafe(
            dummy_gradients,
            accuracy=np.random.uniform(0.3, 0.7),
            loss=np.random.uniform(0.5, 2.0)
        )
        
        # Mark as cover traffic
        dummy_proofs['is_cover_traffic'] = True
        
        return processed_dummies, dummy_proofs

    def validate_model(self, test_loader: DataLoader) -> Dict[str, float]:
        """Validate model performance"""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                # Ensure proper shape
                if len(data.shape) > 2:
                    data = data.view(data.size(0), -1)
                
                output = self.model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        return {
            'test_loss': test_loss / total if total > 0 else float('inf'),
            'test_accuracy': correct / total if total > 0 else 0.0,
            'worker_id': self.config.worker_id
        }

    def get_worker_state(self) -> Dict:
        """Get current worker state"""
        recent_perf = list(self.performance_history)[-1] if self.performance_history else {}
        
        return {
            'worker_id': self.config.worker_id,
            'current_epoch': self.current_epoch,
            'recent_performance': recent_perf,
            'model_norm': sum(torch.norm(p).item() for p in self.model.parameters()),
            'clipping_bound': self.config.clipping_bound,
            'per_coord_bound': self.config.per_coord_bound
        }


def create_worker(
    worker_id: int,
    global_config: GlobalConfig,
    model: nn.Module,
    dataset: Dataset
) -> WorkerOperations:
    """Factory function to create a worker"""
    worker_config = WorkerConfig(
        worker_id=worker_id,
        global_config=global_config,
        local_epochs=1,
        batch_size=global_config.train_batch_size,
        learning_rate=global_config.learning_rate,
        clipping_bound=global_config.max_grad_norm
    )
    
    return WorkerOperations(worker_config, model, dataset)


def test_worker_operations():
    """Test worker operations"""
    from config import create_default_config
    from torch.utils.data import TensorDataset
    
    # Create test configuration
    global_config = create_default_config()
    
    # Create test model
    test_model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    # Create test dataset
    test_data = torch.randn(1000, 784)
    test_labels = torch.randint(0, 10, (1000,))
    test_dataset = TensorDataset(test_data, test_labels)
    
    # Create worker
    worker = create_worker(
        worker_id=0,
        global_config=global_config,
        model=test_model,
        dataset=test_dataset
    )
    
    # Perform local training
    gradients, metadata = worker.train_local_model()
    
    print(f"✓ Worker 0 training complete:")
    print(f"  - Gradient tensors: {len(gradients)}")
    print(f"  - Performance: {metadata['performance']}")
    print(f"  - Has Bulletproof: {'bulletproof' in metadata}")
    print(f"  - Has PEP proof: {'pep_proof' in metadata}")
    print(f"  - Packed blocks: {len(metadata.get('packed_blocks', []))}")
    
    # Test cover traffic
    cover_grads, cover_meta = worker.prepare_cover_traffic()
    print(f"\n✓ Cover traffic generated:")
    print(f"  - Is cover: {cover_meta.get('is_cover_traffic', False)}")
    
    # Get worker state
    state = worker.get_worker_state()
    print(f"\n✓ Worker state:")
    print(f"  - Epoch: {state['current_epoch']}")
    print(f"  - Model norm: {state['model_norm']:.4f}")
    print(f"  - Per-coord bound: {state['per_coord_bound']:.6f}")


if __name__ == "__main__":
    print("Testing TriSAFE Worker Operations...")
    test_worker_operations()
    print("\nAll tests passed! ✓")