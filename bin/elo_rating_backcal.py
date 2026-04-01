import numpy as np
from statistics import NormalDist

LOG10 = np.log(10.0)
ELO_SCALE = 400.0
S = LOG10 / ELO_SCALE  # logistic <-> Elo conversion


def _fit_elo_impl(
    win_rates,
    games=None,
    names=None,
    base_rating=1500.0,
    eps=1e-6,
    max_iter=100,
    tol=1e-9,
    confidence=None,
):
    P = np.asarray(win_rates, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("win_rates must be a square matrix")

    n = P.shape[0]

    if names is None:
        names = [f"player_{i}" for i in range(n)]
    if len(names) != n:
        raise ValueError("len(names) must match matrix size")

    # 片側しか入っていない場合にも対応しつつ、p_ij と p_ji を整合化する
    P_sym = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        P_sym[i, i] = np.nan
        for j in range(i + 1, n):
            vals = []
            if not np.isnan(P[i, j]):
                vals.append(P[i, j])
            if not np.isnan(P[j, i]):
                vals.append(1.0 - P[j, i])

            if not vals:
                raise ValueError(f"missing win rate for pair ({i}, {j})")

            p = float(np.mean(vals))
            # 0% / 100% は理論上 Elo 差が無限大になるので、少しだけ丸める
            p = np.clip(p, eps, 1.0 - eps)

            P_sym[i, j] = p
            P_sym[j, i] = 1.0 - p

    # 重み（試合数）
    if games is None:
        W = np.ones((n, n), dtype=float)
    else:
        G = np.asarray(games, dtype=float)
        if G.shape == ():
            W = np.full((n, n), float(G))
        elif G.shape == (n, n):
            W = np.zeros((n, n), dtype=float)
            for i in range(n):
                for j in range(i + 1, n):
                    vals = []
                    if not np.isnan(G[i, j]):
                        vals.append(G[i, j])
                    if not np.isnan(G[j, i]):
                        vals.append(G[j, i])

                    if not vals:
                        raise ValueError(f"missing games count for pair ({i}, {j})")

                    w = float(np.mean(vals))
                    W[i, j] = W[j, i] = max(w, 0.0)
        else:
            raise ValueError("games must be None, a scalar, or an NxN matrix")

    np.fill_diagonal(W, 0.0)

    def loss_grad_hess(r):
        """
        目的関数:
            sum_{i<j} w_ij * CE( observed_p_ij , model_p_ij )

        model_p_ij = 1 / (1 + 10^(-(r_i-r_j)/400))
        """
        loss = 0.0
        g = np.zeros(n, dtype=float)
        H = np.zeros((n, n), dtype=float)

        for i in range(n):
            for j in range(i + 1, n):
                w = W[i, j]
                if w == 0:
                    continue

                y = P_sym[i, j]            # 実測勝率
                z = S * (r[i] - r[j])      # logistic の内部変数
                z_clip = np.clip(z, -50.0, 50.0)
                p = 1.0 / (1.0 + np.exp(-z_clip))

                # soft label y に対する binary cross entropy
                loss += w * (np.logaddexp(0.0, z) - y * z)

                e = w * (p - y) * S
                h = w * p * (1.0 - p) * S * S

                g[i] += e
                g[j] -= e

                H[i, i] += h
                H[j, j] += h
                H[i, j] -= h
                H[j, i] -= h

        return loss, g, H

    # 全員に同じ定数を足しても勝率は変わらないので、最後の1人を 0 に固定する
    r = np.zeros(n, dtype=float)
    free = np.arange(n - 1)

    for _ in range(max_iter):
        loss, g, H = loss_grad_hess(r)

        Hf = H[np.ix_(free, free)] + 1e-12 * np.eye(n - 1)
        gf = g[free]

        try:
            step_dir = np.linalg.solve(Hf, gf)
        except np.linalg.LinAlgError:
            step_dir = np.linalg.lstsq(Hf, gf, rcond=None)[0]

        # ニュートン法 + 簡単なバックトラック
        step = 1.0
        improved = False
        while step >= 1e-6:
            candidate = r.copy()
            candidate[free] -= step * step_dir
            new_loss, _, _ = loss_grad_hess(candidate)
            if new_loss <= loss:
                r = candidate
                improved = True
                break
            step *= 0.5

        if not improved:
            break

        if np.linalg.norm(step * step_dir) < tol:
            break

    _, _, H_final = loss_grad_hess(r)

    # 平均を base_rating に合わせる
    r_centered = r - np.mean(r) + base_rating
    ratings = {name: float(rating) for name, rating in zip(names, r_centered)}

    if confidence is None:
        return ratings

    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0, 1)")

    Hf = H_final[np.ix_(free, free)] + 1e-12 * np.eye(n - 1)
    try:
        cov_free = np.linalg.inv(Hf)
    except np.linalg.LinAlgError:
        cov_free = np.linalg.pinv(Hf)

    cov_fixed = np.zeros((n, n), dtype=float)
    cov_fixed[np.ix_(free, free)] = cov_free

    # r_centered = (I - 11^T/n) r_fixed + const
    C = np.eye(n) - np.ones((n, n), dtype=float) / float(n)
    cov_centered = C @ cov_fixed @ C.T

    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    ci_half_width = {}
    for i, name in enumerate(names):
        var = max(float(cov_centered[i, i]), 0.0)
        ci_half_width[name] = float(z * np.sqrt(var))

    return ratings, ci_half_width


def fit_elo_from_winrates(
    win_rates,
    games=None,
    names=None,
    base_rating=1500.0,
    eps=1e-6,
    max_iter=100,
    tol=1e-9,
):
    """
    win_rates[i, j] = プレイヤー i が j に勝つ実測勝率 (0.0 ~ 1.0)
    - 対角成分は無視
    - 下三角は 1 - 上三角でもよいし、片側だけ埋めて他方を np.nan にしてもよい
    games[i, j] = その組み合わせの試合数（重み）
    - None なら全ペア同じ重み
    - 試合数が多いペアほど、推定で重く扱われる

    戻り値: {name: rating}
    """

    return _fit_elo_impl(
        win_rates,
        games=games,
        names=names,
        base_rating=base_rating,
        eps=eps,
        max_iter=max_iter,
        tol=tol,
        confidence=None,
    )


def fit_elo_from_winrates_with_interval(
    win_rates,
    games=None,
    names=None,
    base_rating=1500.0,
    eps=1e-6,
    max_iter=100,
    tol=1e-9,
    confidence=0.95,
):
    """
    fit_elo_from_winrates と同じ推定に加え、各プレイヤーの信頼区間の半幅を返す。

    戻り値: (ratings_dict, ci_half_width_dict)
      - ratings_dict[name] = 推定レーティング
      - ci_half_width_dict[name] = 指定信頼水準の半幅（例: 95%なら +- の値）
    """

    return _fit_elo_impl(
        win_rates,
        games=games,
        names=names,
        base_rating=base_rating,
        eps=eps,
        max_iter=max_iter,
        tol=tol,
        confidence=confidence,
    )


def expected_win_matrix(ratings, names=None):
    """
    推定されたレーティングから、理論上の相互勝率行列を返す
    """
    if isinstance(ratings, dict):
        if names is None:
            names = list(ratings.keys())
        r = np.array([ratings[name] for name in names], dtype=float)
    else:
        r = np.asarray(ratings, dtype=float)
        if names is None:
            names = [f"player_{i}" for i in range(len(r))]

    d = r[:, None] - r[None, :]
    P = 1.0 / (1.0 + 10.0 ** (-d / 400.0))
    np.fill_diagonal(P, np.nan)
    return names, P