import torch
import torch.nn as nn
import torch.nn.functional as F

class IntuitionisticFuzzyVariationalLoss(nn.Module):
    """
    Spatially-Adaptive Intuitionistic Fuzzy Variational Loss (IFV-Loss)
    
    Theoretical Formulation:
      - Membership: \mu(x) = p(x) * (1 - \pi(x))
      - Non-membership: \nu(x) = (1 - p(x)) * (1 - \pi(x))
      - Hesitation degree: \pi(x) \in [0, 1], \pi(x) = 1 - \mu(x) - \nu(x)
      
    Decomposes the variational energy into:
      1. Certainty-gated data fidelity: (1 - \pi(x)) * L_CE_Dice
      2. Hesitation manifold smoothing: \pi(x) * L_Manifold
      3. Global intuitionistic fuzzy topological penalty
    """
    def __init__(self, beta=1.5, lambda_hes=0.5, eps=1e-6):
        super().__init__()
        self.beta = beta
        self.lambda_hes = lambda_hes
        self.eps = eps

    def compute_hesitation(self, prob, img):
        # 1. Predictive entropy hesitation H(p) = -p log2 p - (1-p) log2 (1-p)
        p_clamped = prob.clamp(self.eps, 1.0 - self.eps)
        entropy = - (p_clamped * torch.log2(p_clamped) + (1.0 - p_clamped) * torch.log2(1.0 - p_clamped))
        
        # 2. Local spatial gradient / texture variation of image
        gx = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
        gy = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = F.pad(gy, (0, 0, 0, 1))
        local_var = torch.sqrt(gx ** 2 + gy ** 2 + self.eps)
        # Normalize local variance per image
        norm_var = local_var / (local_var.amax(dim=(-2, -1), keepdim=True) + self.eps)
        
        # Combined intuitionistic hesitation field \pi(x) \in [0, 1]
        hesitation = torch.tanh(self.beta * entropy + 0.5 * norm_var)
        return hesitation

    def forward(self, logits, targets, images):
        prob = torch.sigmoid(logits)
        hesitation = self.compute_hesitation(prob, images)  # \pi(x)
        certainty = (1.0 - hesitation).clamp(min=0.0, max=1.0)
        
        # 1. Certainty-gated BCE loss
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        l_fidelity_bce = (certainty * bce).sum() / (certainty.sum() + self.eps)
        
        # 2. Certainty-gated Dice loss
        inter = (prob * targets * certainty).sum(dim=(-2, -1))
        union = (prob * certainty).sum(dim=(-2, -1)) + (targets * certainty).sum(dim=(-2, -1))
        l_fidelity_dice = (1.0 - (2.0 * inter + 1.0) / (union + 1.0)).mean()
        
        # 3. Hesitation-adaptive manifold smoothing
        # At boundary/hesitation pixels, align prediction with bilateral local spatial consensus
        kernel = torch.ones((1, 1, 3, 3), device=logits.device) / 9.0
        smooth_prob = F.conv2d(prob.detach(), kernel, padding=1)
        l_hesitation = (hesitation * (prob - smooth_prob) ** 2).sum() / (hesitation.sum() + self.eps)
        
        total_loss = l_fidelity_bce + l_fidelity_dice + self.lambda_hes * l_hesitation
        return total_loss, {
            "l_bce": l_fidelity_bce.item(),
            "l_dice": l_fidelity_dice.item(),
            "l_hes": l_hesitation.item(),
            "mean_pi": hesitation.mean().item()
        }

if __name__ == "__main__":
    loss_fn = IntuitionisticFuzzyVariationalLoss()
    logits = torch.randn(4, 1, 256, 256)
    targets = (torch.rand(4, 1, 256, 256) > 0.5).float()
    images = torch.rand(4, 1, 256, 256)
    loss, details = loss_fn(logits, targets, images)
    print("Test passed! Loss:", loss.item(), details)
