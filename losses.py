import torch
import torch.nn.functional as F

def loss_recon(X_synthetic: torch.Tensor, X_current: torch.Tensor) -> torch.Tensor:
    """
    Reconstruction Loss (MSE)
    Ensures structural fidelity between synthetic & real current mammograms.
    """
    return F.mse_loss(X_synthetic, X_current)

def loss_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL Divergence Loss
    Forces the latent space to follow N(0, I).
    Formula: L_KL = -1/2 * Σ(1 + logσ² - μ² - σ²)
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

def loss_tumor_bce(T_hat: torch.Tensor, M_gt: torch.Tensor) -> torch.Tensor:
    """
    Tumor Loss (Binary Cross Entropy)
    Supervises tumor region predictions when ground truth masks are available.
    """
    return F.binary_cross_entropy(T_hat, M_gt)

def gan_terms(D, X_prior, X_current, X_synthetic, epsilon: float):
    """
    GAN Loss (Adversarial)
    Discriminator tries to distinguish real vs synthetic triplets.
    Generator tries to fool the discriminator.

    Inputs:
        D          -> Discriminator
        X_prior    -> Prior mammogram
        X_current  -> Current mammogram
        X_synthetic-> Generated synthetic mammogram
        epsilon    -> Stability constant for log terms

    Returns:
        y_real     -> Discriminator predictions on real triplets
        y_fake     -> Discriminator predictions on fake triplets
        L_GAN      -> Generator's adversarial loss
        L_D        -> Discriminator's loss
    """
    # Real triplet: [X_prior, X_current, X_current]
    x_real = torch.cat([X_prior, X_current, X_current], dim=1)
    y_real = D(x_real).clamp(epsilon, 1 - epsilon)

    # Fake triplet: [X_prior, X_current, X_synthetic]
    x_fake = torch.cat([X_prior, X_current, X_synthetic], dim=1)
    y_fake = D(x_fake).clamp(epsilon, 1 - epsilon)

    # Adversarial loss
    L_GAN = (torch.log(y_real) + torch.log(1.0 - y_fake)).mean()

    # Discriminator's loss
    L_D = -L_GAN

    return y_real, y_fake, L_GAN, L_D