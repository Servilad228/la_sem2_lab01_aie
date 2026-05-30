# algorithms/tt_svd.py

"""
TT-SVD алгоритм: разложение плотного тензора в TT-формат.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


def tt_svd(
    tensor: DenseTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-10
) -> TTTensor:
    """
    Возвращает TTTensor — тензор в TT-формате.
    """
    shape = backend.shape(tensor)
    d = len(shape)

    # 1. Граничный случай: тензор порядка 1 (вектор)
    if d == 1:
        n_1 = shape[0]
        G_1 = backend.reshape(tensor, (1, n_1, 1))
        return TTTensor([G_1])

    # 3. Вычисляем локальный порог усечения
    norm_A = backend.norm(tensor)
    if norm_A > 1e-30:
        delta = (eps / math.sqrt(d - 1)) * norm_A
    else:
        delta = 0.0

    # 2. Инициализация
    C = backend.copy(tensor)
    r_prev = 1
    cores = []

    # 4. Основной цикл (от 1 до d-1 в терминах математики, 0 до d-2 в Python)
    for k in range(d - 1):
        n_k = shape[k]
        
        # Развёртка: C_(k) размера (r_{k-1} * n_k) x (n_{k+1} * ... * n_d)
        C_size = backend.size(C)
        rows = r_prev * n_k
        cols = C_size // rows
        
        C_mat = backend.reshape(C, (rows, cols))

        # SVD: вычисляем сингулярное разложение
        U, S, Vt = backend.svd(C_mat, full_matrices=False)

        # Выбор ранга r_k
        r_k = _compute_truncated_rank(S, delta, max_rank)

        # Усечение
        U_trunc = _truncate_columns(U, r_k, backend)
        S_trunc = _truncate_vector(S, r_k, backend)
        Vt_trunc = _truncate_rows(Vt, r_k, backend)

        # Ядро G_k
        G_k = backend.reshape(U_trunc, (r_prev, n_k, r_k))
        cores.append(G_k)

        # Остаток: обновляем C для следующего шага
        C = _multiply_diag_matrix(S_trunc, Vt_trunc, r_k, backend)
        r_prev = r_k

    # 5. Последнее ядро G_d
    G_d = backend.reshape(C, (r_prev, shape[-1], 1))
    cores.append(G_d)

    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _compute_truncated_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    """
    Возвращает ранг усечения по сингулярным значениям.
    """
    k = S.shape[0]
    if k == 0:
        return 1

    s_max = S[0]
    threshold = max(1e-12, 1e-8 * s_max)

    # Определяем числовой ранг (отсекаем машинный ноль)
    r_hat = 0
    for i in range(k):
        if S[i] > threshold:
            r_hat += 1
        else:
            break

    r_k = r_hat
    
    # Усечение по порогу delta
    if delta > 0.0:
        sum_sq = 0.0
        # Идем с конца и суммируем квадраты отбрасываемых чисел
        for i in range(r_hat - 1, -1, -1):
            val = S[i]
            if sum_sq + val * val <= delta * delta:
                sum_sq += val * val
                r_k -= 1
            else:
                break

    if max_rank is not None:
        r_k = min(r_k, max_rank)

    # Ранг не может быть меньше 1
    return max(1, r_k)


def _truncate_columns(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank столбцов исходной матрицы.
    """
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
    """
    Возвращает матрицу, составленную из первых rank строк исходной матрицы.
    """
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
    """
    Возвращает вектор, состоящий из первых rank элементов исходного вектора.
    """
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
    """
    Возвращает произведение диагональной матрицы на обычную матрицу.
    """
    _, n = backend.shape(matrix)
    result = backend.zeros((rank, n))
    for i in range(rank):
        val = backend.get_element(diag_vec, (i,))
        for j in range(n):
            mat_val = backend.get_element(matrix, (i, j))
            backend.set_element(result, (i, j), val * mat_val)
    return result