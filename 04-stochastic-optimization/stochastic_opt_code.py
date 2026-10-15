"""
MNIST Stochastic Optimization - Complete Module
Implements: Data loading, Model, Optimizers (SGD, Adam, Adagrad), Training pipeline, Visualization
"""

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import time
import copy
import os
import json
from datetime import datetime

# CONFIGURATION

RANDOM_SEED = 42

def set_seed(seed=RANDOM_SEED):
    """Set random seed for reproducibility across all libraries."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# DATA LOADING

def get_data(val_split=0.2, batch_size=64):
    """Load MNIST dataset and split into train/validation loaders."""
    print("Downloading/Loading MNIST data...")
    data_full = torchvision.datasets.MNIST('./data', download=True)
    
    # Handle different torchvision versions
    try:
        train_data = data_full.train_data
        train_labels = data_full.train_labels
    except AttributeError:
        train_data = data_full.data
        train_labels = data_full.targets

    # Normalize to [0,1] and add channel dimension
    x_train = train_data.float() / 255.0
    x_train = x_train.unsqueeze(1)

    full_dataset = torch.utils.data.TensorDataset(x_train, train_labels)
    
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    # Use generator with seed for reproducible splits
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    print(f"Data Split: {train_size} training, {val_size} validation")
    
    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(RANDOM_SEED)),
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False))


# MODEL

class MultinomialLogisticRegression(nn.Module):
    """Multinomial Logistic Regression: 784 inputs -> 10 classes (no bias)."""
    
    def __init__(self, input_dim=784, num_classes=10):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes, bias=False)

    def forward(self, x):
        return self.linear(x.view(x.size(0), -1))


# OPTIMIZERS

class BaseOptimizer:
    """Base class for optimizers with L2 regularization (alpha)."""
    
    def __init__(self, params, lr=0.01, alpha=0.0):
        self.params = list(params)
        self.lr = lr
        self.alpha = alpha

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def step(self):
        raise NotImplementedError


class SGD(BaseOptimizer):
    """Vanilla SGD: w = w - lr * (grad + alpha*w)"""
    
    def step(self):
        with torch.no_grad():
            for p in self.params:
                if p.grad is not None:
                    p -= self.lr * (p.grad + self.alpha * p)


class Adagrad(BaseOptimizer):
    """Adagrad: Adapts learning rate per-parameter based on gradient history."""
    
    def __init__(self, params, lr=0.01, epsilon=1e-8, alpha=0.0):
        super().__init__(params, lr, alpha)
        self.epsilon = epsilon
        self.cache = [torch.zeros_like(p) for p in self.params]

    def step(self):
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is not None:
                    grad = p.grad + self.alpha * p
                    self.cache[i] += grad ** 2
                    p -= (self.lr / (torch.sqrt(self.cache[i]) + self.epsilon)) * grad


class Adam(BaseOptimizer):
    """Adam: Combines momentum and adaptive learning rates with bias correction."""
    
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, alpha=0.0):
        super().__init__(params, lr, alpha)
        self.beta1, self.beta2 = beta1, beta2
        self.epsilon = epsilon
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is not None:
                    grad = p.grad + self.alpha * p
                    self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                    self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
                    m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                    v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                    p -= self.lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)


OPTIMIZERS = {
    "SGD": SGD,
    "Adam": Adam,
    "Adagrad": Adagrad
}


# TRAINING

def evaluate(model, val_loader, device):
    """Evaluate model on validation set. Returns (loss, accuracy%)."""
    criterion = nn.CrossEntropyLoss()
    model.eval()
    correct, total, val_loss = 0, 0, 0.0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return val_loss / len(val_loader), 100 * correct / total


def train_one_epoch(model, train_loader, optimizer, device, alpha=0.0):
    """Train one epoch. Returns (ce_loss, obj_loss, accuracy%, grad_norm)."""
    criterion = nn.CrossEntropyLoss()
    model.train()
    running_ce_loss, correct, total = 0.0, 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        batch_size = labels.size(0)

        optimizer.zero_grad()
        # Let model.forward handle the reshape (consistent with evaluate)
        outputs = model(inputs)
        ce_loss = criterion(outputs, labels)
        
        # Simplified gradient: probs[i, y_i] -= 1, then grad = probs.T @ X / B
        X = inputs.view(batch_size, -1)  # Flatten for gradient computation
        probs = torch.softmax(outputs, dim=1)
        probs[range(batch_size), labels] -= 1
        grad_w = probs.t() @ X / batch_size
        
        # Always use copy_ for stability
        if model.linear.weight.grad is None:
            model.linear.weight.grad = grad_w.clone()
        else:
            model.linear.weight.grad.copy_(grad_w)
        
        optimizer.step()

        running_ce_loss += ce_loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_size
        correct += (predicted == labels).sum().item()
    
    # Compute losses
    avg_ce_loss = running_ce_loss / len(train_loader)
    reg_term = (alpha / 2) * (model.linear.weight.norm() ** 2).item()
    obj_loss = avg_ce_loss + reg_term
    
    # Compute gradient norm for convergence check
    grad_norm = model.linear.weight.grad.norm().item() if model.linear.weight.grad is not None else 0.0
        
    return avg_ce_loss, obj_loss, 100 * correct / total, grad_norm


def train_and_validate(optimizer_class, lr, alpha, train_loader, val_loader, 
                       device, epochs=100, mode='final', 
                       conv_tol=1e-6, conv_patience=5):
    """
    Full training loop with configurable stopping criteria.
    """
    print(f"\n--- Training {optimizer_class.__name__} (LR={lr:.5f}, Alpha={alpha}) [{mode}] ---")
    
    model = MultinomialLogisticRegression().to(device)
    optimizer = optimizer_class(model.parameters(), lr=lr, alpha=alpha)

    # History tracking
    history = {
        'ce_losses': [], 'obj_losses': [], 'train_accs': [], 'val_accs': [],
        'train_times': [], 'eval_times': [], 'total_times': [], 'grad_norms': [],
        'best_epoch': 0, 'best_val_acc': 0.0, 'best_train_loss': float('inf'),
        'convergence_reason': None
    }
    
    best_model_state = None
    patience_counter = 0
    prev_obj_loss = float('inf')
    cumulative_time = 0.0

    for epoch in range(epochs):
        # Training
        train_start = time.time()
        ce_loss, obj_loss, t_acc, grad_norm = train_one_epoch(
            model, train_loader, optimizer, device, alpha=alpha
        )
        train_time = time.time() - train_start
        
        # Validation
        eval_start = time.time()
        _, v_acc = evaluate(model, val_loader, device)
        eval_time = time.time() - eval_start
        
        cumulative_time += train_time + eval_time
        
        # Record history
        history['ce_losses'].append(ce_loss)
        history['obj_losses'].append(obj_loss)
        history['train_accs'].append(t_acc)
        history['val_accs'].append(v_acc)
        history['train_times'].append(train_time)
        history['eval_times'].append(eval_time)
        history['total_times'].append(cumulative_time)
        history['grad_norms'].append(grad_norm)
        
        print(f"Epoch [{epoch+1}/{epochs}] Time: {cumulative_time:.2f}s | "
              f"CE: {ce_loss:.4f} F_α: {obj_loss:.4f} | "
              f"Train: {t_acc:.2f}% Val: {v_acc:.2f}% | ‖∇‖: {grad_norm:.6f}")
        
        # Track best validation
        if v_acc > history['best_val_acc']:
            history['best_val_acc'] = v_acc
            history['best_epoch'] = epoch + 1
            history['best_train_loss'] = obj_loss
            best_model_state = copy.deepcopy(model.state_dict())
        
        # Stopping criteria
        if mode == 'tuning':
            # For tuning (inner loop): stop based on TRAINING OBJECTIVE convergence
            rel_change = abs(prev_obj_loss - obj_loss) / (abs(prev_obj_loss) + 1e-10)
            converging = (rel_change < conv_tol) or (grad_norm < 0.01)
            if converging:
                patience_counter += 1
                if patience_counter >= conv_patience:
                    history['convergence_reason'] = f'Training converged (ΔF/F<{conv_tol} or ‖∇‖<0.01)'
                    print(f"Converged at Epoch {epoch+1}: {history['convergence_reason']}")
                    break
            else:
                patience_counter = 0
            prev_obj_loss = obj_loss
        else:
            # For final: stop when validation stops improving
            if v_acc <= history['best_val_acc'] and epoch > history['best_epoch']:
                patience_counter += 1
                if patience_counter >= 10:
                    history['convergence_reason'] = 'Validation early stopping (patience=10)'
                    print(f"Early stopping at Epoch {epoch+1}")
                    break
            else:
                patience_counter = 0
    
    if history['convergence_reason'] is None:
        history['convergence_reason'] = f'Max epochs ({epochs}) reached'
    
    # Restore best model for final mode
    if mode == 'final' and best_model_state is not None:
        model.load_state_dict(best_model_state)
        _, final_val_acc = evaluate(model, val_loader, device)
        history['final_val_acc'] = final_val_acc
        print(f"Restored best weights (Epoch {history['best_epoch']}). Val: {final_val_acc:.2f}%")
    else:
        history['final_val_acc'] = history['val_accs'][-1] if history['val_accs'] else 0.0
    
    print(f"Finished. Best Val: {history['best_val_acc']:.2f}% at Epoch {history['best_epoch']}")
    return history


def tune_jointly(train_loader, val_loader, device):
    """Two-stage hyperparameter search: coarse then fine."""
    print("\n=== Two-Stage Hyperparameter Search ===")
    print("Stage 1: Coarse search (logspace, 15 epochs, aggressive early stop)")
    print("Stage 2: Fine search around best (30 epochs)")
    
    # Logspace grids for coarse search
    sgd_lrs_coarse = list(np.logspace(-2, 0, 5))      # [0.01, ..., 1.0]
    adam_lrs_coarse = list(np.logspace(-4, -1, 4))    # [1e-4, ..., 0.1]
    alphas_coarse = [0.0] + list(np.logspace(-4, -1, 4))  # [0, 1e-4, ..., 0.1]
    
    best_params = {}
    tuning_history = {}
    
    for name, opt_class in OPTIMIZERS.items():
        print(f"\n{'='*50}")
        print(f"Tuning {name}...")
        tuning_history[name] = {'coarse': [], 'fine': []}
        
        # Stage 1: Coarse Search
        print(f"\n[Stage 1] Coarse search...")
        current_lrs = sgd_lrs_coarse if name == "SGD" else adam_lrs_coarse
        print(f"  LRs: {[f'{x:.4f}' for x in current_lrs]}")
        print(f"  Alphas: {[f'{x:.4f}' for x in alphas_coarse]}")
        
        coarse_best_acc, coarse_best = 0.0, (current_lrs[len(current_lrs)//2], 0.0)
        
        for lr in current_lrs:
            for alpha in alphas_coarse:
                history = train_and_validate(
                    opt_class, lr=lr, alpha=alpha, 
                    train_loader=train_loader, val_loader=val_loader, 
                    device=device, epochs=15, mode='tuning',
                    conv_tol=1e-4, conv_patience=3  # More aggressive
                )
                
                val_acc = history['final_val_acc']
                if np.isnan(val_acc) or val_acc < 11.0:
                    val_acc = 0.0
                
                tuning_history[name]['coarse'].append({
                    'lr': lr, 'alpha': alpha, 'val_acc': val_acc,
                    'epochs': len(history['obj_losses'])
                })
                
                if val_acc > coarse_best_acc:
                    coarse_best_acc = val_acc
                    coarse_best = (lr, alpha)
                    print(f"    > Coarse best: LR={lr:.4f}, α={alpha:.4f} -> {val_acc:.2f}%")
        
        print(f"  Coarse winner: LR={coarse_best[0]:.4f}, α={coarse_best[1]:.4f} ({coarse_best_acc:.2f}%)")
        
        # Stage 2: Fine Search
        print(f"\n[Stage 2] Fine search around (LR={coarse_best[0]:.4f}, α={coarse_best[1]:.4f})...")
        
        # Generate fine grid around coarse best
        lr_center, alpha_center = coarse_best
        fine_lrs = [lr_center * f for f in [0.5, 0.7, 1.0, 1.4, 2.0]]
        if alpha_center > 0:
            fine_alphas = [alpha_center * f for f in [0.5, 0.7, 1.0, 1.4, 2.0]]
        else:
            fine_alphas = [0.0, 1e-5, 1e-4, 5e-4, 1e-3]
        
        print(f"  Fine LRs: {[f'{x:.5f}' for x in fine_lrs]}")
        print(f"  Fine Alphas: {[f'{x:.5f}' for x in fine_alphas]}")
        
        fine_best_acc, fine_best = coarse_best_acc, coarse_best
        
        for lr in fine_lrs:
            for alpha in fine_alphas:
                history = train_and_validate(
                    opt_class, lr=lr, alpha=alpha, 
                    train_loader=train_loader, val_loader=val_loader, 
                    device=device, epochs=30, mode='tuning',
                    conv_tol=1e-5, conv_patience=3
                )
                
                val_acc = history['final_val_acc']
                if np.isnan(val_acc) or val_acc < 11.0:
                    val_acc = 0.0
                
                tuning_history[name]['fine'].append({
                    'lr': lr, 'alpha': alpha, 'val_acc': val_acc,
                    'epochs': len(history['obj_losses'])
                })
                
                if val_acc > fine_best_acc:
                    fine_best_acc = val_acc
                    fine_best = (lr, alpha)
                    print(f"    > Fine best: LR={lr:.5f}, α={alpha:.5f} -> {val_acc:.2f}%")
        
        print(f"\n★ {name} Final: LR={fine_best[0]:.5f}, α={fine_best[1]:.5f} ({fine_best_acc:.2f}%)")
        best_params[name] = {
            'lr': fine_best[0],
            'alpha': fine_best[1],
            'best_val_acc': fine_best_acc
        }
        
    return best_params, tuning_history


# VISUALIZATION

COLORS = {
    'SGD': '#1f77b4',       # Blue
    'Adam': '#d62728',      # Red
    'Adagrad': '#2ca02c'    # Green
}

LINE_STYLES = {
    'SGD': '-',
    'Adam': '--',
    'Adagrad': '-.'
}

MARKERS = {
    'SGD': 'o',
    'Adam': 's',
    'Adagrad': '^'
}


def setup_plot_style():
    """Configure matplotlib for clean academic styling."""
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.labelcolor': 'black',
        'axes.titlecolor': 'black',
        'text.color': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'grid.color': '#cccccc',
        'grid.alpha': 0.7,
        'grid.linestyle': '--',
        'legend.facecolor': 'white',
        'legend.edgecolor': 'black',
        'legend.labelcolor': 'black',
        'font.family': 'serif',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'lines.linewidth': 1.5,
        'lines.antialiased': True,
        'axes.linewidth': 1.0,
        'grid.linewidth': 0.5,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.spines.top': True,
        'axes.spines.right': True
    })


def save_plots(results, output_dir='results'):
    """Save comparison plots with clean academic styling."""
    os.makedirs(output_dir, exist_ok=True)
    setup_plot_style()
    
    # === Cross-Entropy Loss ===
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, data in results.items():
        epochs = range(1, len(data['ce_losses']) + 1)
        ax.plot(epochs, data['ce_losses'], label=name, 
                color=COLORS.get(name, 'gray'),
                linestyle=LINE_STYLES.get(name, '-'),
                marker=MARKERS.get(name, 'o'),
                markersize=4, markevery=max(1, len(epochs)//10))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('Cross-Entropy Loss (CE)')
    ax.legend(loc='best', framealpha=1.0)
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'ce_loss_comparison.png'), bbox_inches='tight')
    plt.close()
    
    # === Objective Loss F_α(w) ===
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, data in results.items():
        epochs = range(1, len(data['obj_losses']) + 1)
        ax.plot(epochs, data['obj_losses'], label=name, 
                color=COLORS.get(name, 'gray'),
                linestyle=LINE_STYLES.get(name, '-'),
                marker=MARKERS.get(name, 'o'),
                markersize=4, markevery=max(1, len(epochs)//10))
    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'Objective $F_\alpha(w) = CE + \frac{\alpha}{2}\|W\|^2$')
    ax.set_title('Training Objective')
    ax.legend(loc='best', framealpha=1.0)
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'obj_loss_comparison.png'), bbox_inches='tight')
    plt.close()

    # === Accuracy Comparison ===
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, data in results.items():
        epochs = range(1, len(data['train_accs']) + 1)
        color = COLORS.get(name, 'gray')
        marker = MARKERS.get(name, 'o')
        # Training accuracy (solid)
        ax.plot(epochs, data['train_accs'], label=f"{name} (Train)", 
                color=color, linestyle='-', marker=marker,
                markersize=4, markevery=max(1, len(epochs)//10))
        # Validation accuracy (dashed)
        if 'val_accs' in data:
            ax.plot(epochs, data['val_accs'], label=f"{name} (Val)", 
                    color=color, linestyle='--', marker=marker,
                    markersize=3, markevery=max(1, len(epochs)//10), alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Training and Validation Accuracy')
    ax.legend(loc='lower right', framealpha=1.0)
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), bbox_inches='tight')
    plt.close()

    # Convergence Speed (Wall-Clock Time)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, data in results.items():
        if 'total_times' in data:
            ax.plot(data['total_times'], data['train_accs'], label=name,
                    color=COLORS.get(name, 'gray'),
                    linestyle=LINE_STYLES.get(name, '-'),
                    marker=MARKERS.get(name, 'o'),
                    markersize=4, markevery=max(1, len(data['total_times'])//10))
    ax.set_xlabel('Wall-Clock Time (s) [train + eval]')
    ax.set_ylabel('Training Accuracy (%)')
    ax.set_title('Convergence Speed')
    ax.legend(loc='best', framealpha=1.0)
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'time_comparison.png'), bbox_inches='tight')
    plt.close()
    
    # Generalization Gap
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, data in results.items():
        if 'val_accs' in data:
            gap = [t - v for t, v in zip(data['train_accs'], data['val_accs'])]
            epochs = range(1, len(gap) + 1)
            ax.plot(epochs, gap, label=name,
                    color=COLORS.get(name, 'gray'),
                    linestyle=LINE_STYLES.get(name, '-'),
                    marker=MARKERS.get(name, 'o'),
                    markersize=4, markevery=max(1, len(epochs)//10))
    ax.axhline(y=0, color='black', linestyle=':', linewidth=1.0, alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gap (%)')
    ax.set_title('Generalization Gap (Train - Val)')
    ax.legend(loc='best', framealpha=1.0)
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'generalization_gap.png'), bbox_inches='tight')
    plt.close()
    
    # Gradient Norm
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, data in results.items():
        if 'grad_norms' in data:
            epochs = range(1, len(data['grad_norms']) + 1)
            ax.plot(epochs, data['grad_norms'], label=name,
                    color=COLORS.get(name, 'gray'),
                    linestyle=LINE_STYLES.get(name, '-'),
                    marker=MARKERS.get(name, 'o'),
                    markersize=4, markevery=max(1, len(epochs)//10))
    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'$\|\nabla F_\alpha\|$')
    ax.set_title('Gradient Norm (Convergence Indicator)')
    ax.set_yscale('log')
    ax.legend(loc='best', framealpha=1.0)
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'gradient_norm.png'), bbox_inches='tight')
    plt.close()

    print(f"Plots saved to: {output_dir}")


# MAIN

def save_results(best_params, tuning_history, final_results, output_dir='results'):
    """Save all training results to JSON files for report generation."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save hyperparameter tuning history
    tuning_file = os.path.join(output_dir, f'tuning_history_{timestamp}.json')
    with open(tuning_file, 'w', encoding='utf-8') as f:
        json.dump(tuning_history, f, indent=2, ensure_ascii=False)
    print(f"Tuning history saved to: {tuning_file}")
    
    # Save best hyperparameters
    params_file = os.path.join(output_dir, f'best_params_{timestamp}.json')
    with open(params_file, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)
    print(f"Best params saved to: {params_file}")
    
    # Save final training results
    training_file = os.path.join(output_dir, f'training_results_{timestamp}.json')
    with open(training_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    print(f"Training results saved to: {training_file}")
    
    # Save a summary report
    summary = {
        'timestamp': timestamp,
        'random_seed': RANDOM_SEED,
        'convergence_criteria': {
            'tuning': 'Training objective convergence (ΔF < 1e-6 for 5 epochs or ‖∇‖ < 1e-4)',
            'final': 'Validation early stopping (patience=10)'
        },
        'summary': {}
    }
    for name in best_params:
        data = final_results[name]
        summary['summary'][name] = {
            'best_lr': best_params[name]['lr'],
            'best_alpha': best_params[name]['alpha'],
            'tuning_val_acc': best_params[name]['best_val_acc'],
            'best_epoch': data['best_epoch'],
            'best_val_acc': data['best_val_acc'],
            'best_train_loss': data['best_train_loss'],
            'final_train_acc': data['train_accs'][-1] if data['train_accs'] else None,
            'final_val_acc': data['final_val_acc'],
            'final_ce_loss': data['ce_losses'][-1] if data['ce_losses'] else None,
            'final_obj_loss': data['obj_losses'][-1] if data['obj_losses'] else None,
            'total_time': data['total_times'][-1] if data['total_times'] else None,
            'epochs_trained': len(data['ce_losses']),
            'convergence_reason': data['convergence_reason']
        }
    
    summary_file = os.path.join(output_dir, f'summary_{timestamp}.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to: {summary_file}")
    
    return summary_file


def main():
    set_seed(RANDOM_SEED)
    os.makedirs('results', exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Random seed: {RANDOM_SEED}")
    
    train_loader, val_loader = get_data(val_split=0.2)
    best_params, tuning_history = tune_jointly(train_loader, val_loader, device)
    
    print("\n=== Final Training (100 Epochs) ===")
    final_results = {}
    
    for name, opt_class in OPTIMIZERS.items():
        if name not in best_params:
            continue
        lr = best_params[name]['lr']
        alpha = best_params[name]['alpha']
        history = train_and_validate(
            opt_class, lr=lr, alpha=alpha, 
            train_loader=train_loader, val_loader=val_loader, 
            device=device, epochs=100, mode='final'
        )
        final_results[name] = history
        final_results[name]['lr'] = lr
        final_results[name]['alpha'] = alpha
    
    # Save all results
    save_results(best_params, tuning_history, final_results)
    
    # Generate plots
    save_plots(final_results)
    
    print("\n=== Experiment Complete ===")
    for name, data in final_results.items():
        print(f"{name}: Best Val={data['best_val_acc']:.2f}% (Epoch {data['best_epoch']}), "
              f"Time={data['total_times'][-1]:.2f}s, {data['convergence_reason']}")


if __name__ == "__main__":
    main()
