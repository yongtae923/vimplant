# -*- coding: utf-8 -*-
"""
VAE 기반 최적화기 V2 (개선된 버전)

베이지안 최적화와 더 유사한 동작을 하도록 설계된 VAE 최적화기입니다.
베이지안 최적화의 early stopping, exploration/exploitation balance 등을 구현했습니다.
안정성과 성능을 개선했습니다.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import pickle
import os
from typing import List, Tuple, Optional, Callable
import warnings
warnings.filterwarnings('ignore')

class VAE(nn.Module):
    """
    Variational Autoencoder for parameter optimization
    베이지안 최적화와 유사한 동작을 위해 설계되었습니다.
    """
    def __init__(self, input_dim: int = 4, latent_dim: int = 2, hidden_dims: List[int] = [64, 32]):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_var = nn.Linear(prev_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())  # Normalize parameters to [0,1]
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

class VAE_Optimizer_V2:
    """
    VAE-based optimizer V2 for finding optimal parameters
    베이지안 최적화와 유사한 동작을 하도록 설계되었습니다.
    안정성과 성능을 개선했습니다.
    """
    def __init__(self, 
                 param_bounds: List[Tuple[float, float]], 
                 latent_dim: int = 2,
                 hidden_dims: List[int] = [64, 32],
                 device: str = 'cpu'):
        
        self.param_bounds = param_bounds
        self.param_dim = len(param_bounds)
        self.latent_dim = latent_dim
        self.device = device
        
        # Initialize VAE
        self.vae = VAE(self.param_dim, latent_dim, hidden_dims).to(device)
        
        # History tracking (베이지안과 유사)
        self.param_history = []
        self.cost_history = []
        self.best_params = None
        self.best_cost = float('inf')
        
        # Training parameters (개선됨)
        self.learning_rate = 5e-4  # 더 안정적인 학습률
        self.batch_size = 16  # 더 작은 배치 크기
        self.epochs_per_update = 50  # 더 적은 에포크
        
        # 베이지안과 유사한 early stopping 파라미터
        self.N = 5  # 베이지안의 N과 동일
        self.delta = 0.2  # 베이지안의 delta와 동일
        self.thresh = 0.05  # 베이지안의 thresh와 동일
        
        # 메모리 관리 (새로 추가)
        self.MAX_HISTORY = 1000
        
        # 학습 안정성을 위한 변수들 (새로 추가)
        self.training_stable = False
        self.min_samples_for_training = 5
        
    def normalize_params(self, params: np.ndarray) -> np.ndarray:
        """Normalize parameters to [0,1] range"""
        normalized = np.zeros_like(params)
        for i, (param, (low, high)) in enumerate(zip(params, self.param_bounds)):
            normalized[i] = (param - low) / (high - low)
        return normalized
    
    def denormalize_params(self, normalized_params: np.ndarray) -> np.ndarray:
        """Denormalize parameters from [0,1] to original range"""
        denormalized = np.zeros_like(normalized_params)
        for i, (norm_param, (low, high)) in enumerate(zip(normalized_params, self.param_bounds)):
            denormalized[i] = norm_param * (high - low) + low
        return denormalized
    
    def train_vae(self, params: np.ndarray, costs: np.ndarray, epochs: Optional[int] = None):
        """Train VAE on parameter-cost pairs (개선됨)"""
        if epochs is None:
            epochs = self.epochs_per_update
        
        if len(params) < self.min_samples_for_training:
            print(f"Not enough samples for training: {len(params)} < {self.min_samples_for_training}")
            return
            
        # Normalize parameters
        normalized_params = np.array([self.normalize_params(p) for p in params])
        
        # Convert to tensors
        params_tensor = torch.FloatTensor(normalized_params).to(self.device)
        costs_tensor = torch.FloatTensor(costs).to(self.device)
        
        # Create dataset
        dataset = TensorDataset(params_tensor, costs_tensor)
        dataloader = DataLoader(dataset, batch_size=min(self.batch_size, len(params)), shuffle=True)
        
        # Optimizer (더 안정적인 설정)
        optimizer = optim.Adam(self.vae.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5)
        
        # Training loop
        self.vae.train()
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_params, batch_costs in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                recon_params, mu, log_var = self.vae(batch_params)
                
                # Reconstruction loss
                recon_loss = nn.MSELoss()(recon_params, batch_params)
                
                # KL divergence loss (적응적 가중치)
                kl_weight = min(0.01, 1.0 / (len(self.param_history) + 1))
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                
                # Total loss
                loss = recon_loss + kl_weight * kl_loss
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            scheduler.step(avg_loss)
            
            # Early stopping for training
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= 10:  # 10 에포크 동안 개선이 없으면 중단
                print(f"Training stopped early at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}')
        
        self.training_stable = True
        print(f"VAE training completed with final loss: {avg_loss:.6f}")
    
    def generate_candidates(self, n_candidates: int = 100, 
                          exploration_factor: float = 1.0,
                          use_best_samples: bool = True) -> np.ndarray:
        """Generate candidate parameters using VAE (개선됨)"""
        candidates = []
        
        # 베이지안과 유사하게 최고 성능 샘플들을 기반으로 후보 생성
        if use_best_samples and len(self.cost_history) > 5:
            # 비용이 낮은 상위 20% 샘플들 선택
            n_best = max(1, len(self.cost_history) // 5)
            best_indices = np.argsort(self.cost_history)[:n_best]
            best_params = [self.param_history[i] for i in best_indices]
            
            # 최고 성능 샘플들을 기반으로 변형된 후보 생성
            for _ in range(n_candidates):
                # 랜덤하게 최고 성능 샘플 선택
                base_params = best_params[np.random.randint(0, len(best_params))]
                
                # 적응적 노이즈 (비용에 따라 조정)
                noise_scale = 0.05 * (1 + self.best_cost)  # 비용이 높을수록 더 큰 노이즈
                noise = np.random.normal(0, noise_scale, len(base_params))
                candidate = base_params + noise
                candidates.append(candidate)
        else:
            # VAE를 사용한 후보 생성 (더 안전하게)
            if self.training_stable and len(self.param_history) >= self.min_samples_for_training:
                try:
                    self.vae.eval()
                    with torch.no_grad():
                        z = torch.randn(n_candidates, self.latent_dim).to(self.device)
                        
                        # Add exploration noise
                        if exploration_factor > 0:
                            noise = torch.randn_like(z) * exploration_factor
                            z = z + noise
                        
                        # Decode to parameter space
                        normalized_candidates = self.vae.decode(z).cpu().numpy()
                        
                        # Denormalize
                        for candidate in normalized_candidates:
                            denorm_candidate = self.denormalize_params(candidate)
                            candidates.append(denorm_candidate)
                except Exception as e:
                    print(f"VAE generation failed: {e}, falling back to random sampling")
                    # VAE 실패 시 랜덤 샘플링으로 폴백
                    for _ in range(n_candidates):
                        random_params = np.array([
                            np.random.uniform(low, high) for low, high in self.param_bounds
                        ])
                        candidates.append(random_params)
            else:
                # VAE가 준비되지 않았거나 안정적이지 않을 때 랜덤 샘플링
                for _ in range(n_candidates):
                    random_params = np.array([
                        np.random.uniform(low, high) for low, high in self.param_bounds
                    ])
                    candidates.append(random_params)
        
        # 파라미터 범위 내로 클리핑
        clipped_candidates = []
        for candidate in candidates:
            clipped = np.clip(candidate, 
                            [bounds[0] for bounds in self.param_bounds],
                            [bounds[1] for bounds in self.param_bounds])
            clipped_candidates.append(clipped)
        
        return np.array(clipped_candidates)
    
    def update_vae(self, new_params: np.ndarray, new_cost: float):
        """Update VAE with new parameter-cost pair (개선됨)"""
        self.param_history.append(new_params)
        self.cost_history.append(new_cost)
        
        # Update best solution
        if new_cost < self.best_cost:
            self.best_cost = new_cost
            self.best_params = new_params.copy()
        
        # 메모리 관리
        if len(self.param_history) > self.MAX_HISTORY:
            # 오래된 데이터 제거 (최근 50%만 유지)
            keep_size = self.MAX_HISTORY // 2
            self.param_history = self.param_history[-keep_size:]
            self.cost_history = self.cost_history[-keep_size:]
            print(f"History truncated to {keep_size} samples")
        
        # Retrain VAE periodically (베이지안과 유사하게)
        if len(self.param_history) % 15 == 0 and len(self.param_history) >= self.min_samples_for_training:
            print(f"Retraining VAE with {len(self.param_history)} samples...")
            self.train_vae(np.array(self.param_history), np.array(self.cost_history))
    
    def custom_stopper(self, N: int = 5, delta: float = 0.2, thresh: float = 0.05) -> bool:
        """
        베이지안 최적화의 custom_stopper와 동일한 로직
        Returns True (stops the optimization) when 
        the difference between best and worst of the best N are below delta AND the best is below thresh
        """
        if len(self.cost_history) >= N:
            func_vals = np.sort(self.cost_history)
            worst = func_vals[N - 1]
            best = func_vals[0]
            
            return (abs((best - worst)/worst) < delta) & (best < thresh)
        else:
            return False
    
    def optimize(self, 
                cost_function: Callable, 
                n_iterations: int = 150, 
                n_candidates_per_iter: int = 50,
                initial_samples: int = 10,
                exploration_decay: float = 0.95) -> Tuple[np.ndarray, float]:
        """
        Main optimization loop (베이지안과 유사한 동작, 개선됨)
        
        Args:
            cost_function: Function that takes parameters and returns cost
            n_iterations: Number of optimization iterations (베이지안과 동일)
            n_candidates_per_iter: Number of candidates to generate per iteration
            initial_samples: Number of initial random samples for VAE training (베이지안과 동일)
            exploration_decay: Factor to reduce exploration over time
        
        Returns:
            Tuple of (best_parameters, best_cost)
        """
        
        print(f"Starting VAE optimization V2 with {n_iterations} iterations...")
        
        # Generate initial random samples (베이지안과 동일)
        print(f"Generating {initial_samples} initial random samples...")
        initial_params = []
        initial_costs = []
        
        for _ in range(initial_samples):
            # Random parameters within bounds
            random_params = np.array([
                np.random.uniform(low, high) for low, high in self.param_bounds
            ])
            
            # Evaluate cost
            try:
                cost = cost_function(*random_params)
                initial_params.append(random_params)
                initial_costs.append(cost)
                print(f"Initial sample: {random_params} -> Cost: {cost:.4f}")
            except Exception as e:
                print(f"Error evaluating initial sample: {e}")
                continue
        
        # Train VAE on initial data
        if len(initial_params) > 0:
            print("Training VAE on initial data...")
            self.train_vae(np.array(initial_params), np.array(initial_costs))
        
        # Main optimization loop
        exploration_factor = 1.0
        no_improvement_count = 0
        
        for iteration in range(n_iterations):
            print(f"\n--- Iteration {iteration + 1}/{n_iterations} ---")
            
            # 베이지안과 유사한 early stopping 체크
            if self.custom_stopper(self.N, self.delta, self.thresh):
                print("Early stopping: custom_stopper condition met")
                break
            
            # Generate candidates (베이지안과 유사한 전략)
            use_best_samples = iteration > 0 and len(self.cost_history) > 5
            candidates = self.generate_candidates(
                n_candidates=n_candidates_per_iter,
                exploration_factor=exploration_factor,
                use_best_samples=use_best_samples
            )
            
            # Evaluate candidates
            best_candidate_cost = float('inf')
            best_candidate_params = None
            
            for i, candidate in enumerate(candidates):
                try:
                    cost = cost_function(*candidate)
                    
                    if cost < best_candidate_cost:
                        best_candidate_cost = cost
                        best_candidate_params = candidate.copy()
                    
                    print(f"Candidate {i+1}: {candidate} -> Cost: {cost:.4f}")
                    
                except Exception as e:
                    print(f"Error evaluating candidate {i+1}: {e}")
                    continue
            
            # Update VAE with best candidate
            if best_candidate_params is not None:
                self.update_vae(best_candidate_params, best_candidate_cost)
                print(f"Best candidate: {best_candidate_params} -> Cost: {best_candidate_cost:.4f}")
                print(f"Best overall: {self.best_params} -> Cost: {self.best_cost:.4f}")
                
                # 개선 여부 체크
                if best_candidate_cost < self.best_cost:
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1
            
            # Decay exploration
            exploration_factor *= exploration_decay
            
            # 베이지안과 유사한 early stopping (개선됨)
            if len(self.cost_history) >= 20:
                recent_costs = self.cost_history[-20:]
                if max(recent_costs) - min(recent_costs) < 0.01:
                    print("Early stopping: no significant improvement")
                    break
            
            # 연속 개선 실패 시 early stopping
            if no_improvement_count >= 30:
                print("Early stopping: no improvement for 30 consecutive iterations")
                break
        
        print(f"\nOptimization completed!")
        print(f"Best parameters: {self.best_params}")
        print(f"Best cost: {self.best_cost:.4f}")
        
        return self.best_params, self.best_cost
    
    def plot_optimization_history(self, save_path=None):
        """Plot optimization history (베이지안과 유사한 시각화)"""
        if len(self.cost_history) == 0:
            print("No optimization history to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Cost history
        ax1.plot(self.cost_history)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Cost')
        ax1.set_title('Cost History (베이지안과 유사)')
        ax1.grid(True)
        
        # Parameter history
        param_history_array = np.array(self.param_history)
        param_names = ['alpha', 'beta', 'offset_from_base', 'shank_length']
        for i in range(self.param_dim):
            ax2.plot(param_history_array[:, i], label=param_names[i])
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Parameter Value')
        ax2.set_title('Parameter History')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # 파일 경로가 제공되면 저장
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Optimization history plot saved to: {save_path}")
        
        # plt.show()  # 저장만 하도록 주석 처리
    
    def save_results(self, filename: str, optimization_time: float = None):
        """Save optimization results"""
        results = {
            'best_params': self.best_params,
            'best_cost': self.best_cost,
            'param_history': self.param_history,
            'cost_history': self.cost_history,
            'param_bounds': self.param_bounds,
            'vae_state_dict': self.vae.state_dict(),
            'training_stable': self.training_stable
        }
        
        if optimization_time is not None:
            results['optimization_time'] = optimization_time
        
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Results saved to {filename}")
    
    def load_results(self, filename: str):
        """Load optimization results"""
        with open(filename, 'rb') as f:
            results = pickle.load(f)
        
        self.best_params = results['best_params']
        self.best_cost = results['best_cost']
        self.param_history = results['param_history']
        self.cost_history = results['cost_history']
        
        if 'vae_state_dict' in results:
            self.vae.load_state_dict(results['vae_state_dict'])
        
        if 'training_stable' in results:
            self.training_stable = results['training_stable']
        
        print(f"Results loaded from {filename}")
    
    def get_optimization_summary(self, optimization_time: float = None):
        """베이지안과 유사한 최적화 요약 정보 반환"""
        if len(self.cost_history) == 0:
            return "No optimization data available"
        
        summary = {
            'total_iterations': len(self.cost_history),
            'best_cost': self.best_cost,
            'best_params': self.best_params,
            'cost_improvement': self.cost_history[0] - self.best_cost,
            'final_cost': self.cost_history[-1],
            'cost_std': np.std(self.cost_history),
            'early_stopping_triggered': self.custom_stopper(self.N, self.delta, self.thresh),
            'training_stable': self.training_stable,
            'memory_usage': len(self.param_history)
        }
        
        if optimization_time is not None:
            summary['optimization_time'] = optimization_time
        
        return summary
