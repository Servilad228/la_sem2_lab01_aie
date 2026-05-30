# algorithms/tt_round.py

"""
TT-округление.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface
from algorithms.canonical_form import right_canonicalize


def tt_round(
    tt: TTTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-10
) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор с уменьшенными рангами
    """
    d = tt.order
    
    # 1. Граничный случай: тензор порядка 1 (вектор)
    if d == 1:
        return tt.copy()

    # 2. Правый проход (полная правая ортогонализация)
    tt_orth = right_canonicalize(tt, backend)
    
    # 3. Вычисляем глобальный порог усечения
    G_1 = tt_orth.cores[0]
    norm_G1 = backend.norm(G_1)
    
    if norm_G1 > 1e-30:
        delta = (eps * norm_G1) / math.sqrt(d - 1)
    else:
        delta = 0.0

    # Копируем ядра для безопасной модификации на левом проходе
    cores = [backend.copy(core) for core in tt_orth.cores]

    # 4. Левый проход (SVD-усечение)
    for k in range(d - 1):
        core = cores[k]
        r_prev, n_k, r_next = backend.shape(core)
        
        # (a) Разворачиваем ядро G_k
        matrix = backend.reshape(core, (r_prev * n_k, r_next))
        
        # (b) Вычисляем сингулярное разложение
        U, S, Vt = backend.svd(matrix, full_matrices=False)
        
        # (c) Выбор нового ранга
        r_new = _compute_rank(S, delta, max_rank)
        
        # Усечение матриц
        U_trunc = _truncate_columns(U, r_new, backend)
        S_trunc = _truncate_vector(S, r_new, backend)
        Vt_trunc = _truncate_rows(Vt, r_new, backend)
        
        # (d) Сворачиваем левые векторы в новое сжатое ядро
        cores[k] = backend.reshape(U_trunc, (r_prev, n_k, r_new))
        
        # (e) Поглощаем остаток в следующее ядро G_{k+1}
        R = _multiply_diag_matrix(S_trunc, Vt_trunc, r_new, backend)
        
        next_core = cores[k + 1]
        _, n_next, r_next_next = backend.shape(next_core)
        
        # Разворачиваем следующее ядро для умножения на матрицу R
        next_matrix = backend.reshape(next_core, (r_next, n_next * r_next_next))
        new_next_matrix = backend.matmul(R, next_matrix)
        
        # Сворачиваем обратно
        cores[k + 1] = backend.reshape(new_next_matrix, (r_new, n_next, r_next_next))

    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _compute_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    """
    Возвращает int ранг усечения по вектору сингулярных значений.
    """
    k = S.shape[0]
    if k == 0:
        return 1

    s_max = S[0]
    threshold = max(1e-12, 1e-8 * s_max)

    r_hat = 0
    for i in range(k):
        if S[i] > threshold:
            r_hat += 1
        else:
            break

    r_new = r_hat
    
    if delta > 0.0:
        sum_sq = 0.0
        for i in range(r_hat - 1, -1, -1):
            val = S[i]
            if sum_sq + val * val <= delta * delta:
                sum_sq += val * val
                r_new -= 1
            else:
                break

    if max_rank is not None:
        r_new = min(r_new, max_rank)

    return max(1, r_new)


def _truncate_columns(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    m, n = backend.shape(matrix)
    r = min(n, rank)
    result = backend.zeros((m, r))
    for i in range(m):
        for j in range(r):
            backend.set_element(result, (i, j), backend.get_element(matrix, (i, j)))
    return result


def _truncate_rows(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    m, n = backend.shape(matrix)
    r = min(m, rank)
    result = backend.zeros((r, n))
    for i in range(r):
        for j in range(n):
            backend.set_element(result, (i, j), backend.get_element(matrix, (i, j)))
    return result


def _truncate_vector(
    vector: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    k = backend.shape(vector)[0]
    r = min(k, rank)
    result = backend.zeros((r,))
    for i in range(r):
        backend.set_element(result, (i,), backend.get_element(vector, (i,)))
    return result


def _multiply_diag_matrix(
    diag_vec: DenseTensor,
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    _, n = backend.shape(matrix)
    result = backend.zeros((rank, n))
    for i in range(rank):
        val = backend.get_element(diag_vec, (i,))
        for j in range(n):
            mat_val = backend.get_element(matrix, (i, j))
            backend.set_element(result, (i, j), val * mat_val)
    return result