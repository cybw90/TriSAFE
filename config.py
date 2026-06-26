"""Configuration for TriSAFE Federated Learning System"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch
import logging
from pathlib import Path

@dataclass
class GlobalConfig:
    """Global configuration matching TriSAFE paper parameters (Table 1)"""
    
    # System parameters
    num_workers: int = 100  # Number of workers
    num_rounds: int = 100   # Training rounds
    seed: int = 42  # Random seed for reproducibility
    
    # Model architecture
    input_dim: int = 784    # Input dimension (e.g., MNIST)
    hidden_dim: int = 128   # Hidden layer dimension
    output_dim: int = 10    # Output dimension (e.g., 10 classes)
    
    # Training parameters
    train_batch_size: int = 32
    test_batch_size: int = 100
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0001
    local_epochs: int = 1
    
    # Privacy parameters (from paper)
    privacy_budget: float = 10.0  # ε (epsilon)
    delta: float = 1e-5          # δ (delta) for DP
    noise_multiplier: float = 0.1  # σ (sigma) for Gaussian noise
    min_epsilon: float = 0.1
    max_epsilon: float = 2.0

    # In GlobalConfig class
    production_mode: bool = False  # Set to False for development/testing
    development_skip_crypto: bool = True  # Skip expensive crypto operations in dev
    
    # Clipping parameters (from paper)
    max_grad_norm: float = 1.0  # C - global clipping bound
    
    # Time window parameters (from paper)
    time_window: float = 300.0  # δ from paper (not 300!)
    cover_traffic_ratio: float = 0.5  # ρ from paper
    dropout_tolerance: float = 0.3  # drop from paper
    
    # Paillier parameters (from paper Table 1)
    paillier_modulus_bits: int = 3072  # N_pai bits
    packing_base_exp: int = 29  # b (packing base exponent)
    slots_per_ciphertext: int = 64  # L (slots per ciphertext)
    fixed_point_scale_exp: int = 16  # log2(S_fp)
    weight_scale_exp: int = 16  # log2(S_α)
    
    # Security parameters (from paper)
    security_margin_bits: int = 128  # λ (security parameter)
    packing_overflow_prob_exp: int = -80  # log2(δ_wrap)
    folding_weight_bits: int = 32  # κ (for PEP folding)
    
    # Bulletproof parameters
    range_proof_bits: int = 32  # m for range [0, 2^m)
    
    # Threshold parameters
    threshold_t: int = 2  # t in t-of-n threshold
    threshold_n: int = 3  # n in t-of-n threshold

    # Add to GlobalConfig class:
    production_mode: bool = False  # Set to True for production deployment
    
    # Byzantine parameters
    byzantine_threshold: float = 0.3  # Fraction of Byzantine workers tolerated
    
    # System optimization parameters
    cluster_frequency: int = 5
    validation_frequency: int = 10
    checkpoint_frequency: int = 10
    evaluation_frequency: int = 5  # Added for compatibility
    
    # Resource parameters
    max_workers_per_round: int = 50
    min_workers_per_round: int = 10
    worker_selection_strategy: str = "random"  # Options: "random", "performance", "trust"
    
    # Data parameters
    data_path: str = "./data"  # Path to data directory
    dataset: str = "mnist"  # Default dataset
    data_distribution: str = "iid"  # Default distribution
    non_iid_degree: float = 0.5  # For non-IID distribution
    train_test_split: float = 0.8  # Train/test split ratio
    validation_split: float = 0.1  # Validation split from training data
    
    # Device configuration
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    # Logging configuration
    log_level: str = "INFO"
    log_frequency: int = 1
    metrics_log_path: str = "metrics.log"
    
    # Threat detection parameters (for compatibility)
    threat_detection_threshold: float = 0.7
    system_constant: float = 1.0
    baseline_threat: float = 0.1
    
    def __post_init__(self):
        """Compute derived parameters and validate configuration"""
        # Create data directory if it doesn't exist
        Path(self.data_path).mkdir(parents=True, exist_ok=True)
        
        # Compute actual scale values from exponents
        self.fixed_point_scale = 2 ** self.fixed_point_scale_exp  # S_fp
        self.weight_scale = 2 ** self.weight_scale_exp  # S_α
        
        # Compute per-coordinate bound
        self.per_coord_bound = self.max_grad_norm / (self.input_dim ** 0.5)
        
        # Compute range proof offset
        self.range_offset = int(self.fixed_point_scale * self.per_coord_bound)
        
        # Validate threshold parameters
        assert self.threshold_t <= self.threshold_n, "t must be <= n in threshold scheme"
        assert self.threshold_t >= 2, "Need at least 2 parties for threshold"
        
        # Validate privacy parameters
        assert 0 < self.privacy_budget <= 10, "Privacy budget must be in (0, 10]"
        assert 0 < self.delta < 1, "Delta must be in (0, 1)"
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def get_phe_config(self) -> Dict[str, Any]:
        """Get PHE-specific configuration"""
        return {
            'modulus_bits': self.paillier_modulus_bits,
            'packing_base_exp': self.packing_base_exp,
            'slots_per_ciphertext': self.slots_per_ciphertext,
            'fixed_point_scale': self.fixed_point_scale,
            'weight_scale': self.weight_scale,
            'security_margin_bits': self.security_margin_bits,
            'folding_weight_bits': self.folding_weight_bits,
            'packing_overflow_prob': 2 ** self.packing_overflow_prob_exp
        }
    
    def get_privacy_config(self) -> Dict[str, Any]:
        """Get privacy-specific configuration"""
        return {
            'epsilon': self.privacy_budget,
            'delta': self.delta,
            'noise_multiplier': self.noise_multiplier,
            'clipping_bound': self.max_grad_norm,
            'min_epsilon': self.min_epsilon,
            'max_epsilon': self.max_epsilon
        }
    
    def get_threshold_config(self) -> Dict[str, Any]:
        """Get threshold scheme configuration"""
        return {
            't': self.threshold_t,
            'n': self.threshold_n,
            'helpers': [f"H{i+1}" for i in range(self.threshold_n)]
        }
    
    def get_time_window_config(self) -> Dict[str, Any]:
        """Get time window and cover traffic configuration"""
        return {
            'time_window': self.time_window,
            'cover_traffic_ratio': self.cover_traffic_ratio,
            'max_delay': self.time_window / 2  # Maximum acceptable delay
        }
    
    def validate_for_production(self) -> bool:
        """Validate configuration for production deployment"""
        issues = []
        
        # Check security parameters
        if self.paillier_modulus_bits < 2048:
            issues.append("Paillier modulus should be at least 2048 bits for production")
        
        if self.security_margin_bits < 128:
            issues.append("Security margin should be at least 128 bits")
        
        # Check privacy parameters
        if self.privacy_budget > 5.0:
            issues.append("Privacy budget may be too large for strong privacy")
        
        # Check system parameters
        if self.num_workers < 10:
            issues.append("Too few workers for robust federated learning")
        
        if self.byzantine_threshold > 0.5:
            issues.append("Byzantine threshold too high - system may not be robust")
        
        if issues:
            for issue in issues:
                logging.warning(f"Configuration issue: {issue}")
            return False
        
        return True


def create_default_config() -> GlobalConfig:
    """Create default configuration matching paper specifications"""
    return GlobalConfig()


def create_test_config() -> GlobalConfig:
    """Create configuration for testing with reduced parameters"""
    return GlobalConfig(
        num_workers=10,
        num_rounds=10,
        paillier_modulus_bits=1024,  # Reduced for testing
        time_window=60.0,  # Shorter window for testing
        train_batch_size=16,
        security_margin_bits=64,  # Reduced for testing
        log_level="DEBUG",
        data_path="./test_data",
        seed=42
    )


def create_production_config() -> GlobalConfig:
    """Create production configuration with enhanced security"""
    config = GlobalConfig(
        num_workers=100,
        num_rounds=200,
        paillier_modulus_bits=4096,  # Enhanced security
        security_margin_bits=256,  # Enhanced security
        privacy_budget=0.5,  # Stronger privacy
        noise_multiplier=2.0,  # More noise for privacy
        byzantine_threshold=0.2,  # More conservative
        time_window=180.0,  # Tighter time window
        log_level="WARNING",
        data_path="./production_data",
        seed=42
    )
    
    if not config.validate_for_production():
        logging.error("Production configuration validation failed")
    
    return config


@dataclass
class ExperimentConfig:
    """Configuration for running experiments"""
    global_config: GlobalConfig
    experiment_name: str
    num_trials: int = 3
    save_checkpoints: bool = True
    checkpoint_dir: str = "./checkpoints"
    results_dir: str = "./results"
    data_dir: str = "./data"
    
    # Attack configurations
    attack_type: Optional[str] = None  # "sign_flip", "noise", "replay"
    attack_fraction: float = 0.0  # Fraction of Byzantine workers
    
    # Dataset configuration
    dataset_name: str = "mnist"  # Options: "mnist", "cifar10", "fashion_mnist"
    data_distribution: str = "iid"  # Options: "iid", "non_iid"
    non_iid_degree: float = 0.5  # For non-IID distribution
    
    # Monitoring configuration
    enable_tensorboard: bool = False
    tensorboard_dir: str = "./runs"
    save_gradients: bool = False
    save_weights: bool = True
    
    def get_attack_config(self) -> Dict[str, Any]:
        """Get attack-specific configuration"""
        return {
            'enabled': self.attack_type is not None,
            'type': self.attack_type,
            'fraction': self.attack_fraction,
            'num_attackers': int(self.global_config.num_workers * self.attack_fraction)
        }


def load_config_from_file(filepath: str) -> GlobalConfig:
    """Load configuration from a JSON or YAML file"""
    import json
    import os
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        if filepath.endswith('.json'):
            config_dict = json.load(f)
        else:
            # Add YAML support if needed
            raise ValueError("Only JSON configuration files are currently supported")
    
    return GlobalConfig(**config_dict)


def save_config_to_file(config: GlobalConfig, filepath: str):
    """Save configuration to a JSON file"""
    import json
    from dataclasses import asdict
    
    config_dict = asdict(config)
    
    # Convert Path objects to strings
    for key, value in config_dict.items():
        if isinstance(value, Path):
            config_dict[key] = str(value)
    
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    logging.info(f"Configuration saved to {filepath}")


if __name__ == "__main__":
    # Test configurations
    print("=== DEFAULT CONFIGURATION (Paper Parameters) ===")
    default_config = create_default_config()
    print(f"Random seed: {default_config.seed}")
    print(f"Data path: {default_config.data_path}")
    print(f"Dataset: {default_config.dataset}")
    print(f"Paillier modulus: {default_config.paillier_modulus_bits} bits")
    print(f"Packing: {default_config.slots_per_ciphertext} slots with base 2^{default_config.packing_base_exp}")
    print(f"Fixed-point scale: 2^{default_config.fixed_point_scale_exp}")
    print(f"Time window: {default_config.time_window}s")
    print(f"Threshold: {default_config.threshold_t}-of-{default_config.threshold_n}")
    print(f"Privacy budget: ε={default_config.privacy_budget}, δ={default_config.delta}")
    
    print("\n=== PHE CONFIGURATION ===")
    phe_config = default_config.get_phe_config()
    for key, value in phe_config.items():
        print(f"  {key}: {value}")
    
    print("\n=== TESTING CONFIGURATION ===")
    test_config = create_test_config()
    print(f"Workers: {test_config.num_workers}")
    print(f"Rounds: {test_config.num_rounds}")
    print(f"Data path: {test_config.data_path}")
    print(f"Paillier modulus: {test_config.paillier_modulus_bits} bits")
    
    print("\n=== PRODUCTION VALIDATION ===")
    prod_config = create_production_config()
    is_valid = prod_config.validate_for_production()
    print(f"Production ready: {is_valid}")
    
    # Save example configuration
    save_config_to_file(default_config, "trisafe_config.json")