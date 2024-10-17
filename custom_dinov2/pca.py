import torch


def torch_pca(features: torch.Tensor, n_components: int = 3):
    """
    Perform PCA using PyTorch.

    Args:
        features (torch.Tensor): Input features of shape (num_samples, num_features).
        n_components (int): Number of principal components to keep.

    Returns:
        torch.Tensor: Projected features of shape (num_samples, n_components).
    """
    # Centering the features
    mean = torch.mean(features, dim=0, keepdim=True)
    centered_features = features - mean

    # Compute covariance matrix (C = X^T * X / (n-1))
    cov_matrix = torch.mm(centered_features.T, centered_features) / (
        centered_features.shape[0] - 1
    )

    # Eigen decomposition
    eigvals, eigvecs = torch.linalg.eigh(cov_matrix)

    # Sort eigenvalues and eigenvectors
    sorted_idx = torch.argsort(eigvals, descending=True)
    eigvecs = eigvecs[:, sorted_idx]

    # Project features onto the top `n_components` principal components
    principal_components = eigvecs[:, :n_components]
    projected_features = torch.mm(centered_features, principal_components)

    return projected_features


def torch_minmax_scale(tensor: torch.Tensor, feature_range=(0, 1)):
    """
    Scales the input tensor to the given feature range [min, max].

    Args:
        tensor (torch.Tensor): Input tensor.
        feature_range (tuple): Desired range of transformed data (default is (0, 1)).

    Returns:
        torch.Tensor: Scaled tensor.
    """
    min_val, max_val = feature_range
    tensor_min = tensor.min()
    tensor_max = tensor.max()

    # Scale the tensor to [0, 1]
    scaled_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)

    # Scale to [min_val, max_val]
    scaled_tensor = scaled_tensor * (max_val - min_val) + min_val

    return scaled_tensor


def dinov2_pca(
    features: torch.Tensor, masks: torch.Tensor = None, fg_min_norm: float = 0.01
):
    bs = features.shape[0]
    features = features.permute(0, 2, 3, 1)
    if masks is not None:
        input_masks = []
        for i in range(bs):
            input_masks.append(masks[i].float())
        input_masks = torch.stack(input_masks).bool().numpy().flatten()
    feat_h, feat_w, feat_c = features.shape[1:]
    features = features.reshape(bs * feat_h * feat_w, feat_c)
    projected_features = torch_pca(features, n_components=3)
    norm_features = torch_minmax_scale(projected_features)
    # segment using the first component
    pca_features_bg = norm_features[:, 0] < fg_min_norm
    if masks is not None:
        pca_features_bg = ~input_masks
    pca_features_fg = ~pca_features_bg

    pca_features_rem = torch_pca(features[pca_features_fg], n_components=3)
    pca_features_rem = torch_minmax_scale(pca_features_rem)

    pca_features_rgb = torch.zeros((features.shape[0], 3)).to(features)
    pca_features_rgb[pca_features_fg] = pca_features_rem.to(features)
    pca_features_rgb = pca_features_rgb.reshape(bs, feat_h, feat_w, 3)

    return pca_features_rgb.permute(0, 3, 1, 2)
