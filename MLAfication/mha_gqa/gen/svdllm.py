import torch

def original_svd(Wk, Wv, input, r):
    W = torch.cat([Wk, Wv])
    U, S, V = torch.linalg.svd(W.float(), full_matrices=False)
    U, S, V = U[:, :r], S[:r], V[:r, :]
    W_up = U @ torch.diag(torch.sqrt(S))
    W_down = torch.diag(torch.sqrt(S)) @ V
    W_up_k = W_up[: Wk.size(0), :]
    W_up_v = W_up[Wk.size(0) :, :]
    return W_up_k.detach().cpu(), W_up_v.detach().cpu(), W_down.detach().cpu()


def svdllm(Wk, Wv, input, r):
    input = input["kvproj_input"]
    Us, Ss, Vs = torch.linalg.svd(input, full_matrices=False)

    W = torch.cat([Wk, Wv])
    Ss_sqrt = torch.diag(torch.sqrt(Ss))
    D = W @ Us @ Ss_sqrt
    # print("D ", D.size())
    Uws, Sws, Vws = torch.linalg.svd(D.float(), full_matrices=False)
    Uws, Sws, Vws = Uws[:, :r], Sws[:r], Vws[:r, :]
    Ss_sqrt_inv = torch.linalg.inv(Ss_sqrt)
    Us_inv = torch.linalg.inv(Us)
    W_up, W_down = Uws, Vws @ Ss_sqrt_inv @ Us_inv

    W_up = W_up @ torch.diag(torch.sqrt(Sws))
    W_down = torch.diag(torch.sqrt(Sws)) @ W_down

    W_up_k = W_up[: Wk.size(0), :]
    W_up_v = W_up[Wk.size(0) :, :]
    return W_up_k.detach().cpu(), W_up_v.detach().cpu(), W_down.detach().cpu()


def scaled_svdllm(Wk, Wv, input, r, alpha=None):
    # div `alpha` for k_nope&value balance
    if alpha is None:
        alpha = (input["k_norm"] + 1e-10) / (input["v_norm"] + 1e-10)
    Wk = Wk / alpha
    # svdllm
    Us, Ss, Vs = torch.linalg.svd(input["kvproj_input"], full_matrices=False)
    Us, Ss, Vs = Us.to(Wk.device), Ss.to(Wk.device), Vs.to(Wk.device)
    W = torch.cat([Wk, Wv])
    Ss_sqrt = torch.diag(torch.sqrt(Ss))
    D = W @ Us @ Ss_sqrt
    Uws, Sws, Vws = torch.linalg.svd(D.float(), full_matrices=False)
    Uws, Sws, Vws = Uws[:, :r], Sws[:r], Vws[:r, :]
    Ss_sqrt_inv = torch.linalg.inv(Ss_sqrt)
    Us_inv = torch.linalg.inv(Us)
    W_up, W_down = Uws, Vws @ Ss_sqrt_inv @ Us_inv
    # up&down balance
    W_up_norm = torch.linalg.vector_norm(W_up, dim=0)
    W_down_norm = torch.linalg.vector_norm(W_down, dim=1)
    scales = W_up_norm / W_down_norm
    W_up = W_up @ torch.diag(torch.rsqrt(scales))
    W_down = torch.diag(torch.sqrt(scales)) @ W_down
    # svdllm
    W_up = W_up @ torch.diag(torch.sqrt(Sws))
    W_down = torch.diag(torch.sqrt(Sws)) @ W_down
    # mul alpha for k_nope&value balance
    W_up_k = alpha * W_up[: Wk.size(0), :]
    W_up_v = W_up[Wk.size(0) :, :]
    return W_up_k.detach().cpu(), W_up_v.detach().cpu(), W_down.detach().cpu()


def scaled_UD_svdllm(Wk, Wv, input, r):
    # svdllm
    Us, Ss, Vs = torch.linalg.svd(input["kvproj_input"], full_matrices=False)
    W = torch.cat([Wk, Wv])
    Ss_sqrt = torch.diag(torch.sqrt(Ss))
    D = W @ Us @ Ss_sqrt
    Uws, Sws, Vws = torch.linalg.svd(D.float(), full_matrices=False)
    Uws, Sws, Vws = Uws[:, :r], Sws[:r], Vws[:r, :]
    Ss_sqrt_inv = torch.linalg.inv(Ss_sqrt)
    Us_inv = torch.linalg.inv(Us)
    W_up, W_down = Uws, Vws @ Ss_sqrt_inv @ Us_inv
    # up&down balance
    W_up_norm = torch.linalg.vector_norm(W_up, dim=0)
    W_down_norm = torch.linalg.vector_norm(W_down, dim=1)
    scales = W_up_norm / W_down_norm
    W_up = W_up @ torch.diag(torch.rsqrt(scales))
    W_down = torch.diag(torch.sqrt(scales)) @ W_down
    # svdllm
    W_up = W_up @ torch.diag(torch.sqrt(Sws))
    W_down = torch.diag(torch.sqrt(Sws)) @ W_down
    # mul alpha for k_nope&value balance
    W_up_k = W_up[: Wk.size(0), :]
    W_up_v = W_up[Wk.size(0) :, :]
    return W_up_k.detach().cpu(), W_up_v.detach().cpu(), W_down.detach().cpu()


def scaled_KV_svdllm(Wk, Wv, input, r):
    # div `alpha` for k_nope&value balance
    alpha = (input["k_norm"] + 1e-10) / (input["v_norm"] + 1e-10)
    Wk = Wk / alpha
    # svdllm
    Us, Ss, Vs = torch.linalg.svd(input["kvproj_input"], full_matrices=False)
    W = torch.cat([Wk, Wv])
    Ss_sqrt = torch.diag(torch.sqrt(Ss))
    D = W @ Us @ Ss_sqrt
    Uws, Sws, Vws = torch.linalg.svd(D.float(), full_matrices=False)
    Uws, Sws, Vws = Uws[:, :r], Sws[:r], Vws[:r, :]
    Ss_sqrt_inv = torch.linalg.inv(Ss_sqrt)
    Us_inv = torch.linalg.inv(Us)
    W_up, W_down = Uws, Vws @ Ss_sqrt_inv @ Us_inv
    # svdllm
    W_up = W_up @ torch.diag(torch.sqrt(Sws))
    W_down = torch.diag(torch.sqrt(Sws)) @ W_down
    # mul alpha for k_nope&value balance
    W_up_k = alpha * W_up[: Wk.size(0), :]
    W_up_v = W_up[Wk.size(0) :, :]
    return W_up_k.detach().cpu(), W_up_v.detach().cpu(), W_down.detach().cpu()
