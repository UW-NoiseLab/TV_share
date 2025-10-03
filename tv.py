import torch
import torch.nn.functional as F
import copy

def _to_bchw(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 2: return t.unsqueeze(0).unsqueeze(0)
    if t.dim() == 3: return t.unsqueeze(0)
    if t.dim() == 4: return t
    raise ValueError(f"Expect [H,W]/[C,H,W]/[B,C,H,W], got {tuple(t.shape)}")

def tv_isotropic(z, tv_eps):
        # z: [B,C,H,W]
        dx = z[..., :, 1:] - z[..., :, :-1]
        dy = z[..., 1:, :] - z[..., :-1, :]
        dx = F.pad(dx, (0,1,0,0))
        dy = F.pad(dy, (0,0,0,1))
        return torch.sqrt(dx*dx + dy*dy + tv_eps*tv_eps).mean()

def reconstruct_tv(
    x_init: torch.Tensor,
    y_pred: torch.Tensor,
    forward_fun,                # callable: forward_fun(x, **forward_kwargs) -> y
    forward_kwargs: dict = None,
    **kwargs
):
    """
    TV-regularized gradient descent reconstruction with bounded output in [0,1].

    Parameters
    ----------
    x_init : torch.Tensor
        Initial guess, [C,H,W] or [B,C,H,W].
    y_pred : torch.Tensor
        Observed image, from forward_fun(x), [B,C,H,W].
    forward_fun : callable
        forward_fun(x, **forward_kwargs).
    forward_kwargs : dict
        Arguments for forward function.
    bound_method : str
        'clamp'  -> project x to [0,1] after each update
        'sigmoid' -> optimize in logit space, always map with sigmoid to [0,1]
    kwargs : dict
        Extra options: tv_lambda, iters, lr, tv_eps, patience, delta_tol, verbose, device.

    Returns
    -------
    x_rec : torch.Tensor
        Reconstructed image, same shape as x_init, values in [0,1].
    history : dict
        Logs of losses {'total': [...], 'data': [...], 'tv': [...]}
    """
    tv_lambda = float(kwargs.get("tv_lambda", 1e-2))
    iters     = int(kwargs.get("iters", 300))
    lr        = float(kwargs.get("lr", 1e-1))
    verbose   = bool(kwargs.get("verbose", True))
    tv_eps    = float(kwargs.get("tv_eps", 1e-3))
    device    = str(kwargs.get("device", "cpu"))
    patience  = kwargs.get("patience", 30)    
    delta_tol = kwargs.get("delta_tol", 1e-2)     
    bound_method = kwargs.get("bound_method", "sigmoid")

    print(f"[TV-Recon] method={bound_method}, tv_lambda={tv_lambda}, iters={iters}, lr={lr}, device={device}")

    forward_kwargs_used = copy.deepcopy(forward_kwargs or {})
    for k in ("photons_per_unit", "noise", "add_noise"):
        forward_kwargs_used.pop(k, None)
    forward_kwargs_used["device"] = device
    y_pred = y_pred.to(device)

    history = {"total": [], "data": [], "tv": []}

    # ----------------- variable init -----------------
    if bound_method == "sigmoid":
        # use z in logit space
        z_init = torch.logit(x_init.clamp(1e-6, 1 - 1e-6))
        var = z_init.detach().clone().to(device).requires_grad_(True)
    else:
        # clamp mode: directly optimize x_rec
        var = x_init.detach().clone().to(device).requires_grad_(True)

    opt = torch.optim.Adam([var], lr=lr)
    no_improve = 0

    for t in range(iters):
        opt.zero_grad(set_to_none=True)

        if bound_method == "sigmoid":
            x_rec = torch.sigmoid(var)
        else:
            x_rec = var

        Ax = forward_fun(x_rec, **forward_kwargs_used)
        data_loss = 0.5 * F.mse_loss(Ax, y_pred)
        tv_loss = tv_isotropic(x_rec, tv_eps)
        total = data_loss + tv_lambda * tv_loss
        total.backward()
        opt.step()

        if bound_method == "clamp":
            with torch.no_grad():
                var.clamp_(0, 1)

        total_val = float(total.detach().cpu())
        data_val  = float(data_loss.detach().cpu())
        tv_val    = float(tv_loss.detach().cpu())
        history["total"].append(total_val)
        history["data"].append(data_val)
        history["tv"].append(tv_val)

        # early stopping
        if t > 3:
            prev = history["total"][-2]
            rel_change = abs(total_val - prev) / max(abs(prev), 1e-18)
            if rel_change < delta_tol or total_val > prev:
                no_improve += 1
                if no_improve >= patience:
                    print(f"[TV-Recon] Early stopping at iter {t} "
                          f"(no improvement for {patience} steps)")
                    break
            else:
                no_improve -=2

        if verbose and (t % max(1, iters // 10) == 0 or t == iters-1):
            print(f"[TV-Recon] iter {t}/{iters}  total={total_val:.4e}  "
                  f"data={data_val:.4e}  tv={tv_val:.4e}")

    # final reconstruction
    if bound_method == "sigmoid":
        x_out = torch.sigmoid(var).detach()
    else:
        x_out = var.detach().clamp(0, 1)

    history['total'] = torch.tensor(history['total'])
    history['data']  = torch.tensor(history['data'])
    history['tv']    = torch.tensor(history['tv'])   

    return x_out, history


