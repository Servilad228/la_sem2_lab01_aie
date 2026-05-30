# algorithms/canonical_form.py

"""
Приведение TT-тензора в канонические формы (полная правая и
левая ортогонализация ядер).
"""

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


def left_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в лево-канонической форме.
    """
    d = tt.order
    cores = [backend.copy(core) for core in tt.cores]
    
    for k in range(d - 1):
        core = cores[k]
        r_prev, n_k, r_next = backend.shape(core)
        
        # 1. Разворачиваем ядро в матрицу: (r_{k-1} * n_k) x r_k
        matrix = backend.reshape(core, (r_prev * n_k, r_next))
        
        # 2. Выполняем QR-разложение
        Q, R = backend.qr(matrix)
        
        # 3. Сворачиваем Q обратно в левоортогональное ядро
        cores[k] = backend.reshape(Q, (r_prev, n_k, r_next))
        
        # 4. Поглощаем матрицу R в следующее ядро G_{k+1}
        next_core = cores[k + 1]
        _, n_next, r_next_next = backend.shape(next_core)
        
        # Для матричного умножения разворачиваем следующее ядро по первой моде
        next_matrix = backend.reshape(next_core, (r_next, n_next * r_next_next))
        new_next_matrix = backend.matmul(R, next_matrix)
        
        # Сворачиваем обратно
        cores[k + 1] = backend.reshape(new_next_matrix, (r_next, n_next, r_next_next))
        
    return TTTensor(cores)


def right_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в право-канонической форме.
    """
    d = tt.order
    cores = [backend.copy(core) for core in tt.cores]
    
    for k in range(d - 1, 0, -1):
        core = cores[k]
        r_prev, n_k, r_next = backend.shape(core)
        
        # 1. Разворачиваем ядро: r_{k-1} x (n_k * r_k)
        matrix = backend.reshape(core, (r_prev, n_k * r_next))
        
        # 2. Выполняем RQ-разложение через QR транспонированной матрицы
        matrix_t = backend.transpose(matrix)
        Q, R = backend.qr(matrix_t)
        
        # matrix = R^T Q^T
        Q_t = backend.transpose(Q)
        R_t = backend.transpose(R)
        
        # 3. Сворачиваем Q^T обратно в правоортогональное ядро
        cores[k] = backend.reshape(Q_t, (r_prev, n_k, r_next))
        
        # 4. Поглощаем R^T в предыдущее ядро G_{k-1}
        prev_core = cores[k - 1]
        r_prev_prev, n_prev, _ = backend.shape(prev_core)
        
        # Разворачиваем предыдущее ядро и умножаем справа на R^T
        prev_matrix = backend.reshape(prev_core, (r_prev_prev * n_prev, r_prev))
        new_prev_matrix = backend.matmul(prev_matrix, R_t)
        
        # Сворачиваем обратно
        cores[k - 1] = backend.reshape(new_prev_matrix, (r_prev_prev, n_prev, r_prev))
        
    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _numerical_rank(
    S: DenseTensor,
    rel_tol: float = 1e-8,
    abs_tol: float = 1e-12
) -> int:
    """
    Возвращает числовой ранг матрицы по вектору сингулярных значений.
    """
    if S.size == 0:
        return 0
    
    s_max = S[0]
    threshold = max(abs_tol, rel_tol * s_max)
    r = 0
    for i in range(S.size):
        if S[i] > threshold:
            r += 1
        else:
            break
    return max(1, r)


def _truncate_columns(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    m, n = backend.shape(matrix)
    rank = min(n, rank)
    result = backend.zeros((m, rank))
    for i in range(m):
        for j in range(rank):
            backend.set_element(result, (i, j), backend.get_element(matrix, (i, j)))
    return result


def _truncate_rows(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    m, n = backend.shape(matrix)
    rank = min(m, rank)
    result = backend.zeros((rank, n))
    for i in range(rank):
        for j in range(n):
            backend.set_element(result, (i, j), backend.get_element(matrix, (i, j)))
    return result


def _truncate_vector(
    vector: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    k = backend.shape(vector)[0]
    rank = min(k, rank)
    result = backend.zeros((rank,))
    for i in range(rank):
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


def _multiply_columns_by_diag(
    matrix: DenseTensor,
    diag_vec: DenseTensor,
    backend: BackendInterface
) -> DenseTensor:
    m, _ = backend.shape(matrix)
    rank = backend.shape(diag_vec)[0]
    result = backend.zeros((m, rank))
    for j in range(rank):
        val = backend.get_element(diag_vec, (j,))
        for i in range(m):
            mat_val = backend.get_element(matrix, (i, j))
            backend.set_element(result, (i, j), mat_val * val)
    return result