

import torch

import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Create a fine grid of logit values (from very confident wrong → very confident right)
logits = np.linspace(-12, 12, 1201)   # 1201 points → smooth curve
logits_t = torch.from_numpy(logits).float()

# Compute BCE loss for both cases (reduction='none' gives per-sample loss)
bce_positive = F.binary_cross_entropy_with_logits(
    logits_t, torch.ones_like(logits_t), reduction='none'
)
bce_negative = F.binary_cross_entropy_with_logits(
    logits_t, torch.zeros_like(logits_t), reduction='none'
)

# Plot
plt.figure(figsize=(11, 6.5))
plt.plot(logits, np.exp(-bce_positive.numpy()), label='True label = 1  (positive class)', linewidth=2.8, color='#1f77b4')
plt.plot(logits, np.exp(-bce_negative.numpy()), label='True label = 0  (negative class)', linewidth=2.8, color='#ff7f0e')

# Helpful annotations
plt.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1.2)
plt.text(-11.5, 11.2, 'Model predicts\nvery negative', ha='left', va='top', fontsize=10, color='0.4')
plt.text(11.5, 11.2, 'Model predicts\nvery positive', ha='right', va='top', fontsize=10, color='0.4')
plt.text(0.4, 6.5, 'Logit = 0\n(p ≈ 0.5)', rotation=90, fontsize=10, color='0.5')

# Zoom-friendly limits
plt.ylim(0, 13)
plt.xlim(-12.5, 12.5)

plt.title('Binary Cross Entropy with Logits\nLoss contribution per sample', fontsize=15, pad=12)
plt.xlabel('Logit (raw model output before sigmoid)', fontsize=12)
plt.ylabel('Loss value', fontsize=12)
plt.legend(fontsize=11, loc='upper center')
plt.grid(True, alpha=0.25, linestyle='--')
plt.tight_layout()

# Optional: mark some example points
examples = [-8, -3, 0, 3, 8]
for x in examples:
    yp = float(bce_positive[abs(logits - x).argmin()])
    yn = float(bce_negative[abs(logits - x).argmin()])
    plt.plot(x, yp, 'o', color='#1f77b4', markersize=6, alpha=0.7)
    plt.plot(x, yn, 'o', color='#ff7f0e', markersize=6, alpha=0.7)

plt.show()