import os
import random
from scipy import stats
from sklearn import logger
from sklearn.decomposition import PCA
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import logging
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import platform 


# Module-level seed worker function for macOS compatibility
def seed_worker(worker_id):
    """Deterministic worker initialization function"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    random.seed(worker_seed)

class FederatedDataset(Dataset):
    """Base class for federated datasets"""
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class DataPreprocessor:
    """Handle data preprocessing operations"""
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, data: np.ndarray) -> None:
        """Fit scaler to data"""
        self.scaler.fit(data)
        self.is_fitted = True
        
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted scaler"""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
        return self.scaler.transform(data)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform data"""
        if not self.is_fitted:
            self.fit(data)
        return self.transform(data)

class FederatedDataLoader:
    """Main data loading and partitioning class"""
    def __init__(self, config):
        self.config = config
        self.preprocessor = DataPreprocessor()
        self.logger = self._setup_logging()
        self.label_mapping = {
            'benign': 0,
            'malicious': 1

        }

    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger('FederatedDataLoader')

##################### Test ##########################

    def load_and_preprocess_data(self, data_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and preprocess data for the configured dataset.

        Dispatches on ``self.config.dataset``:
          - 'edge_iiot' : Edge-IIoTset CSV (binary 'Attack_label')      -> LP.csv
          - 'nbaiot'    : N-BaIoT CSV (multi-class, label auto-detected) -> extracted_features.csv
          - 'mnist' / 'cifar10' : downloaded via torchvision (flattened)
        Returns (features: FloatTensor [N, D], labels: LongTensor [N]).
        """
        # Reproducibility
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)

        dataset = getattr(self.config, 'dataset', 'edge_iiot').lower()
        self.logger.info(f"Preparing dataset: {dataset}")

        if dataset in ('edge_iiot', 'edge-iiotset', 'edge', 'lp'):
            return self._load_edge_iiot(data_path)
        elif dataset in ('nbaiot', 'n-baiot', 'n_baiot'):
            return self._load_nbaiot(data_path)
        elif dataset in ('mnist', 'cifar10', 'cifar'):
            return self._load_image_dataset('cifar10' if dataset == 'cifar' else dataset)
        else:
            raise ValueError(
                f"Unsupported dataset '{dataset}'. "
                f"Use one of: edge_iiot, nbaiot, mnist, cifar10."
            )

    # ---- dataset-specific loaders ------------------------------------- #
    def _resolve_csv(self, data_path: str, default_name: str) -> str:
        """Find a CSV by trying data_path (file or dir), then the CWD."""
        import os
        candidates = []
        if data_path and str(data_path).lower().endswith('.csv'):
            candidates.append(data_path)
        if data_path:
            candidates.append(os.path.join(str(data_path), default_name))
        candidates.append(default_name)
        for c in candidates:
            if os.path.isfile(c):
                return c
        raise FileNotFoundError(
            f"Could not find '{default_name}' (looked in: {candidates}). "
            f"Place the dataset CSV in one of those locations or set config.data_path."
        )

    def _load_edge_iiot(self, data_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Edge-IIoTset: binary intrusion detection ('Attack_label')."""
        csv_path = self._resolve_csv(data_path, 'LP.csv')
        self.logger.info(f"Loading Edge-IIoTset from: {csv_path}")
        data = pd.read_csv(csv_path, low_memory=False)
        if data.empty:
            raise ValueError("Empty dataset")
        if 'Attack_label' not in data.columns:
            raise ValueError("Edge-IIoTset expects an 'Attack_label' column")

        data.fillna(data.mean(numeric_only=True), inplace=True)
        for col in data.select_dtypes(include=['object']).columns:
            if col == 'Attack_label':
                continue
            try:
                data[col] = pd.to_numeric(data[col], errors='coerce')
                mode_val = data[col].mode()[0] if not data[col].mode().empty else 0
                data[col] = data[col].fillna(mode_val)
            except Exception as e:
                self.logger.warning(f"Column '{col}' conversion issue: {e}")
                data[col] = pd.factorize(data[col])[0]
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.fillna(data.mean(numeric_only=True), inplace=True)

        # Deterministically balance the two classes
        class_0 = data[data['Attack_label'] == 0]
        class_1 = data[data['Attack_label'] == 1]
        min_size = min(len(class_0), len(class_1))
        if min_size == 0:
            raise ValueError("One or both classes have zero samples")
        rs = np.random.RandomState(self.config.seed)
        i0 = rs.choice(class_0.index, size=min_size, replace=False)
        i1 = rs.choice(class_1.index, size=min_size, replace=False)
        balanced = pd.concat([class_0.loc[i0], class_1.loc[i1]])
        balanced = balanced.sample(frac=1, random_state=self.config.seed)

        labels = balanced['Attack_label'].values
        features = balanced.drop(columns=['Attack_label', 'Attack_type'], errors='ignore').values
        return self._finalize_tabular(features, labels)

    def _load_nbaiot(self, data_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """N-BaIoT: IoT botnet detection. Label column auto-detected."""
        csv_path = self._resolve_csv(data_path, 'extracted_features.csv')
        self.logger.info(f"Loading N-BaIoT from: {csv_path}")
        data = pd.read_csv(csv_path, low_memory=False)
        if data.empty:
            raise ValueError("Empty dataset")

        label_col = None
        for cand in ['label', 'Label', 'class', 'Class', 'y', 'target', 'Target',
                     'attack', 'Attack', 'category', 'Category', 'type', 'Type']:
            if cand in data.columns:
                label_col = cand
                break
        if label_col is None:
            label_col = data.columns[-1]
            self.logger.warning(f"No known label column found; using last column '{label_col}'")

        data.fillna(data.mean(numeric_only=True), inplace=True)
        labels, _ = pd.factorize(pd.Series(data[label_col].values))  # encode to 0..C-1

        features = data.drop(columns=[label_col], errors='ignore')
        for col in features.select_dtypes(include=['object']).columns:
            features[col] = pd.to_numeric(features[col], errors='coerce')
        features = features.fillna(0.0).values
        return self._finalize_tabular(features, labels)

    def _finalize_tabular(self, features, labels) -> Tuple[torch.Tensor, torch.Tensor]:
        """Clip, scale, and convert tabular features/labels to tensors."""
        features = np.asarray(features, dtype=np.float64)
        features = np.clip(features, -1e10, 1e10)
        features = self.preprocessor.fit_transform(features)
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(np.asarray(labels).astype(np.int64))
        self.logger.info(
            f"Final shapes - Features: {tuple(features.shape)}, Labels: {tuple(labels.shape)}"
        )
        return features, labels

    def _load_image_dataset(self, name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load MNIST / CIFAR-10 via torchvision and flatten to vectors."""
        try:
            from torchvision import datasets as tv_datasets, transforms
        except Exception as e:
            raise ImportError(
                "torchvision is required for image datasets (mnist/cifar10). "
                "Install it with `pip install torchvision`."
            ) from e
        import os
        root = self.config.data_path if self.config.data_path else './data'
        os.makedirs(root, exist_ok=True)
        tfm = transforms.ToTensor()  # pixels -> [0, 1]

        if name == 'mnist':
            train = tv_datasets.MNIST(root, train=True, download=True, transform=tfm)
            test = tv_datasets.MNIST(root, train=False, download=True, transform=tfm)
        else:  # cifar10
            train = tv_datasets.CIFAR10(root, train=True, download=True, transform=tfm)
            test = tv_datasets.CIFAR10(root, train=False, download=True, transform=tfm)

        def to_tensors(ds):
            xs, ys = [], []
            for x, y in ds:
                xs.append(x.view(-1))
                ys.append(int(y))
            return torch.stack(xs), torch.LongTensor(ys)

        x_tr, y_tr = to_tensors(train)
        x_te, y_te = to_tensors(test)
        features = torch.cat([x_tr, x_te], dim=0).float()
        labels = torch.cat([y_tr, y_te], dim=0).long()
        self.logger.info(
            f"Loaded {name}: Features {tuple(features.shape)}, Labels {tuple(labels.shape)}"
        )
        return features, labels
#####################################################

    def _distribute_iid(self, features: torch.Tensor, labels: torch.Tensor) -> List[FederatedDataset]:
        """Distribute data in IID manner with deterministic behavior"""
        try:
            num_clients = self.config.num_workers
            features_np = features.numpy()
            labels_np = labels.numpy()

            # Create deterministic random number generator
            rng = np.random.RandomState(self.config.seed)

            # Get indices for each class
            class_indices = {
                label: np.where(labels_np == label)[0]
                for label in np.unique(labels_np)
            }
            
            # Calculate samples per client per class
            total_samples = len(labels_np)
            samples_per_client = total_samples // num_clients
            
            datasets = []
            remaining_indices = {label: indices.copy() for label, indices in class_indices.items()}

            for i in range(num_clients):
                client_indices = []
                
                # For each class, take proportional samples
                for label, indices in class_indices.items():
                    # Calculate number of samples for this class
                    class_samples = len(indices)
                    client_class_samples = class_samples // num_clients
                    
                    # Handle remainder for last client
                    if i == num_clients - 1:
                        client_class_samples = len(remaining_indices[label])
                    
                    # Deterministically select indices
                    if len(remaining_indices[label]) >= client_class_samples:
                        selected = rng.choice(
                            remaining_indices[label],
                            client_class_samples,
                            replace=False
                        )
                        remaining_indices[label] = np.setdiff1d(
                            remaining_indices[label],
                            selected,
                            assume_unique=True
                        )
                        client_indices.extend(selected)
                
                # Create client dataset with deterministic shuffling
                client_features = torch.FloatTensor(features_np[client_indices])
                client_labels = torch.LongTensor(labels_np[client_indices])
                
                # Deterministic shuffle
                shuffle_idx = rng.permutation(len(client_features))
                client_features = client_features[shuffle_idx]
                client_labels = client_labels[shuffle_idx]
                
                datasets.append(FederatedDataset(client_features, client_labels))
                
                # Log distribution
                unique, counts = np.unique(client_labels.numpy(), return_counts=True)
                distribution = dict(zip(unique, counts))
                total_client_samples = len(client_labels)
                
                self.logger.info(f"\nClient {i} Summary:")
                self.logger.info(f"Total samples: {total_client_samples}")
                for label in np.unique(labels_np):
                    count = distribution.get(label, 0)
                    class_name = [k for k, v in self.label_mapping.items() if v == label][0]
                    percentage = (count / total_client_samples) * 100
                    self.logger.info(f"  {class_name}: {count} samples ({percentage:.2f}%)")
            
            return datasets
                
        except Exception as e:
            self.logger.error(f"Error in IID distribution: {str(e)}")
            raise

#################################### Test ###########################################

    def _distribute_non_iid(self, features: torch.Tensor, labels: torch.Tensor) -> List[FederatedDataset]:
        """Distribute data in Non-IID manner with controlled class imbalance"""
        try:
            num_clients = self.config.num_workers
            features_np = features.numpy()
            labels_np = labels.numpy()
            
            # Adjusted configuration for better balance
            # alpha = 1.0  # Increased for more balanced but still non-IID distribution
            # min_class_ratio = 0.30  # Minimum 35% for any class
            # max_class_ratio = 0.70  # Maximum 65% for any class

            alpha = 0.55  # Increased for more balanced but still non-IID distribution
            min_class_ratio = 0.25  # Minimum 35% for any class
            max_class_ratio = 0.75  # Maximum 65% for any class
            
            unique_classes = np.unique(labels_np)
            num_classes = len(unique_classes)
            
            # Calculate samples per client
            total_samples = len(labels_np)
            base_samples_per_client = total_samples // num_clients
            
            # Initialize client data structures
            client_features = [[] for _ in range(num_clients)]
            client_labels = [[] for _ in range(num_clients)]
            
            # Calculate Dirichlet distribution for each client
            client_class_distributions = np.random.dirichlet(
                [alpha] * num_classes, 
                size=num_clients
            )
            
            # Adjust distributions to prevent extreme imbalance
            client_class_distributions = np.clip(
                client_class_distributions, 
                min_class_ratio,  
                max_class_ratio
            )
            
            # Normalize after clipping
            client_class_distributions = client_class_distributions / client_class_distributions.sum(axis=1, keepdims=True)
            
            # Create class indices lookup
            class_indices = {
                label: np.where(labels_np == label)[0]
                for label in unique_classes
            }
            
            # Distribute data to clients
            for client_id in range(num_clients):
                client_samples = base_samples_per_client
                if client_id == num_clients - 1:  # Handle remainder
                    client_samples = total_samples - (num_clients - 1) * base_samples_per_client
                
                # Calculate target samples for each class
                class_samples = {
                    label: int(client_samples * client_class_distributions[client_id][i])
                    for i, label in enumerate(unique_classes)
                }
                
                # Select samples for each class
                for class_label, target_samples in class_samples.items():
                    available_indices = class_indices[class_label]
                    
                    if len(available_indices) >= target_samples:
                        selected_indices = np.random.choice(
                            available_indices,
                            size=target_samples,
                            replace=False
                        )
                        # Update available indices
                        class_indices[class_label] = np.setdiff1d(available_indices, selected_indices)
                    else:
                        # If not enough samples, use replacement
                        selected_indices = np.random.choice(
                            available_indices,
                            size=target_samples,
                            replace=True
                        )
                    
                    client_features[client_id].extend(features_np[selected_indices])
                    client_labels[client_id].extend([class_label] * len(selected_indices))
            
            # Create and verify datasets
            datasets = []
            for client_id in range(num_clients):
                # Convert to numpy arrays
                client_features_np = np.array(client_features[client_id])
                client_labels_np = np.array(client_labels[client_id])
                
                # Shuffle data
                shuffle_idx = np.random.permutation(len(client_features_np))
                client_features_tensor = torch.FloatTensor(client_features_np[shuffle_idx])
                client_labels_tensor = torch.LongTensor(client_labels_np[shuffle_idx])
                
                datasets.append(FederatedDataset(client_features_tensor, client_labels_tensor))
                
                # Log distribution
                total_samples = len(client_labels_np)
                self.logger.info(f"\nClient {client_id} Summary (Non-IID):")
                self.logger.info(f"Total samples: {total_samples}")
                for label in unique_classes:
                    count = np.sum(client_labels_np == label)
                    class_name = [k for k, v in self.label_mapping.items() if v == label][0]
                    percentage = (count / total_samples) * 100
                    self.logger.info(f"  {class_name}: {count} samples ({percentage:.2f}%)")
            
            return datasets
            
        except Exception as e:
            self.logger.error(f"Error in Non-IID distribution: {str(e)}")
            raise

###############################################################################
    def create_federated_datasets(self, features: torch.Tensor,
                                labels: torch.Tensor,
                                distribution_type: str = 'iid') -> List[FederatedDataset]:
        """Split data into federated datasets"""
        try:
            min_samples_required = self.config.num_workers * 50
            if len(features) < min_samples_required:
                raise ValueError(f"Insufficient data: {len(features)} samples, need at least {min_samples_required}")

            if distribution_type == 'iid':
                self.logger.info("Using IID data distribution")
                return self._distribute_iid(features, labels)
            else:
                self.logger.info("Using Non-IID data distribution")
                return self._distribute_non_iid(features, labels)

        except Exception as e:
            self.logger.error(f"Error creating federated datasets: {str(e)}")
            raise


    def create_train_val_test_split(self, dataset: Dataset,
                                val_split: float = 0.1,
                                test_split: float = 0.1) -> Tuple[Dataset, Dataset, Dataset]:
        """Split dataset with stratified sampling"""
        try:
            total_size = len(dataset)
            all_labels = dataset.labels.numpy()  # Get all labels
            
            # First split: train vs (val+test)
            temp_split = 1 - (val_split + test_split)
            train_idx, temp_idx = self._stratified_split(all_labels, temp_split)
            
            # Second split: val vs test
            val_test_labels = all_labels[temp_idx]
            ratio_val = val_split / (val_split + test_split)
            val_idx_temp, test_idx_temp = self._stratified_split(val_test_labels, ratio_val)
            
            # Convert temp indices to original indices
            val_idx = temp_idx[val_idx_temp]
            test_idx = temp_idx[test_idx_temp]
            
            # Create datasets using indices
            train_dataset = torch.utils.data.Subset(dataset, train_idx)
            val_dataset = torch.utils.data.Subset(dataset, val_idx)
            test_dataset = torch.utils.data.Subset(dataset, test_idx)
            
            # Log split sizes and class distributions
            self.logger.info(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}, Test size: {len(test_idx)}")
            
            return train_dataset, val_dataset, test_dataset
        except Exception as e:
            self.logger.error(f"Error creating train/val/test split: {str(e)}")
            raise


    def _stratified_split(self, labels: np.ndarray, ratio: float) -> Tuple[np.ndarray, np.ndarray]:
        """Helper function for stratified splitting"""
        from sklearn.model_selection import StratifiedShuffleSplit
        
        splitter = StratifiedShuffleSplit(
            n_splits=1, 
            train_size=ratio,
            random_state=self.config.seed
        )
        train_idx, test_idx = next(splitter.split(np.zeros(len(labels)), labels))
        return train_idx, test_idx


    def get_data_loaders(self, dataset: Dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
        """Create deterministic data loader for CPU training"""
        try:
            # Create deterministic generator
            generator = torch.Generator()
            generator.manual_seed(self.config.seed)
                    
            # Adjust batch size based on number of workers
            if self.config.num_workers >= 20:
                adjusted_batch_size = 128  # Smaller batch size for 20+ workers
            else:
                adjusted_batch_size = 256  # Original size for fewer workers
                
            cpu_count = os.cpu_count()
            
            # Platform-specific adjustments
            if platform.system() == 'Darwin':  # macOS
                # On macOS, use fewer workers to avoid multiprocessing issues
                if self.config.num_workers >= 20:
                    num_workers = min(2, cpu_count - 1 if cpu_count else 0)
                else:
                    num_workers = min(4, cpu_count - 1 if cpu_count else 0)
                persistent_workers = num_workers > 0
            else:  # Linux/Ubuntu
                # Original settings for Linux
                if self.config.num_workers >= 20:
                    num_workers = min(cpu_count - 1 if cpu_count else 2, 4)
                else:
                    num_workers = min(cpu_count - 1 if cpu_count else 2, 6)
                persistent_workers = True
                    
            return DataLoader(
                dataset,
                batch_size=adjusted_batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                worker_init_fn=seed_worker if num_workers > 0 else None,  # Use module-level function
                generator=generator,
                pin_memory=False,  # False for CPU
                persistent_workers=persistent_workers,
                drop_last=False
            )
        except Exception as e:
            self.logger.error(f"Error creating data loader: {str(e)}")
            raise



    # def get_data_loaders(self, dataset: Dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    #     """Create deterministic data loader for CPU training"""
    #     try:
    #         # Deterministic worker initialization
    #         # def seed_worker(worker_id):
    #         #     worker_seed = self.config.seed + worker_id
    #         #     np.random.seed(worker_seed)
    #         #     torch.manual_seed(worker_seed)
    #         #     random.seed(worker_seed)



    #         # Create deterministic generator
    #         generator = torch.Generator()
    #         generator.manual_seed(self.config.seed)
                    
    #         # Adjust batch size based on number of workers
    #         if self.config.num_workers >= 20:
    #             adjusted_batch_size = 128  # Smaller batch size for 20+ workers
    #         else:
    #             adjusted_batch_size = 256  # Original size for fewer workers
                
    #         cpu_count = os.cpu_count()
    #         # Adjust num_workers for DataLoader based on total workers
    #         if self.config.num_workers >= 20:
    #             num_workers = min(cpu_count - 1 if cpu_count else 2, 4)  # Reduced to 4 for many workers
    #         else:
    #             num_workers = min(cpu_count - 1 if cpu_count else 2, 6)  # Original setting
                    
    #         return DataLoader(
    #             dataset,
    #             batch_size=adjusted_batch_size,
    #             shuffle=shuffle,
    #             num_workers=num_workers,
    #             worker_init_fn=seed_worker,
    #             generator=generator,
    #             pin_memory=False,  # False for CPU
    #             persistent_workers=True,
    #             drop_last=False
    #         )
    #     except Exception as e:
    #         self.logger.error(f"Error creating data loader: {str(e)}")
    #         raise


def create_federated_data_loaders(config, distribution_type='iid'):
    """Create federated data loaders with specified distribution type"""
    loader = FederatedDataLoader(config)
    features, labels = loader.load_and_preprocess_data(config.data_path)
    
    federated_datasets = loader.create_federated_datasets(
        features, 
        labels, 
        distribution_type=distribution_type
    )
    
    # Create global train/val/test split for evaluation
    global_dataset = FederatedDataset(features, labels)
    train_dataset, val_dataset, test_dataset = loader.create_train_val_test_split(global_dataset)
    
    # Create data loaders
    worker_loaders = [
        loader.get_data_loaders(dataset, config.train_batch_size)
        for dataset in federated_datasets
    ]
    
    val_loader = loader.get_data_loaders(val_dataset, config.test_batch_size, shuffle=False)
    test_loader = loader.get_data_loaders(test_dataset, config.test_batch_size, shuffle=False)
    
    return worker_loaders, val_loader, test_loader

if __name__ == "__main__":
    from config import create_default_config
    
    # Create config
    config = create_default_config()
    
    print("\n" + "="*50)
    print("Testing N_BaIoT Dataset Loader")
    print("="*50)
    
    # Test both IID and Non-IID distributions
    print("\nTesting IID distribution:")
    print("-"*30)
    worker_loaders_iid, val_loader, test_loader = create_federated_data_loaders(config, distribution_type='iid')
    
    print("\nIID Distribution Summary:")
    print(f"Number of workers: {len(worker_loaders_iid)}")
    print(f"Validation batch size: {val_loader.batch_size}")
    print(f"Test batch size: {test_loader.batch_size}")
    
    print("\nTesting Non-IID distribution:")
    print("-"*30)
    worker_loaders_non_iid, val_loader, test_loader = create_federated_data_loaders(config, distribution_type='non-iid')
    
    print("\nNon-IID Distribution Summary:")
    print(f"Number of workers: {len(worker_loaders_non_iid)}")
    print(f"Validation batch size: {val_loader.batch_size}")
    print(f"Test batch size: {test_loader.batch_size}")
    
    # Test data iteration
    print("\nTesting data iteration:")
    print("-"*30)
    
    # Test a worker loader
    worker_loader = worker_loaders_iid[0]
    batch_features, batch_labels = next(iter(worker_loader))
    print(f"\nWorker batch shape: Features {batch_features.shape}, Labels {batch_labels.shape}")
    
    # Test validation loader
    val_features, val_labels = next(iter(val_loader))
    print(f"Validation batch shape: Features {val_features.shape}, Labels {val_labels.shape}")
    
    # Test test loader
    test_features, test_labels = next(iter(test_loader))
    print(f"Test batch shape: Features {test_features.shape}, Labels {test_labels.shape}")
    
    print("\nDataset loading and distribution tests completed successfully!")