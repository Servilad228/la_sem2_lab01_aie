# algorithms/tensor_operations.py

"""
Базовые операции с TT-тензорами.

Все операции работают напрямую с TT-ядрами,
не восстанавливая полный тензор.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


Number = int | float


def tt_add(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат поэлементного сложения двух TT-тензоров.
    """
    d = tt1.order
    cores = []
    
    for k in range(d):
        A = tt1.cores[k]
        B = tt2.cores[k]
        rA_prev, n, rA_next = backend.shape(A)
        rB_prev, _, rB_next = backend.shape(B)
        
        if k == 0:
            # Первое ядро: конкатенация по столбцам (горизонтально)
            C = backend.zeros((1, n, rA_next + rB_next))
            for i in range(n):
                for j in range(rA_next):
                    backend.set_element(C, (0, i, j), backend.get_element(A, (0, i, j)))
                for j in range(rB_next):
                    backend.set_element(C, (0, i, rA_next + j), backend.get_element(B, (0, i, j)))
                    
        elif k == d - 1:
            # Последнее ядро: конкатенация по строкам (вертикально)
            C = backend.zeros((rA_prev + rB_prev, n, 1))
            for i in range(rA_prev):
                for j in range(n):
                    backend.set_element(C, (i, j, 0), backend.get_element(A, (i, j, 0)))
            for i in range(rB_prev):
                for j in range(n):
                    backend.set_element(C, (rA_prev + i, j, 0), backend.get_element(B, (i, j, 0)))
                    
        else:
            # Промежуточные ядра: блочно-диагональная структура
            C = backend.zeros((rA_prev + rB_prev, n, rA_next + rB_next))
            for i in range(rA_prev):
                for j in range(n):
                    for m in range(rA_next):
                        backend.set_element(C, (i, j, m), backend.get_element(A, (i, j, m)))
            for i in range(rB_prev):
                for j in range(n):
                    for m in range(rB_next):
                        backend.set_element(C, (rA_prev + i, j, rA_next + m), backend.get_element(B, (i, j, m)))
                        
        cores.append(C)
        
    return TTTensor(cores)


def tt_scalar_mul(
    tt: TTTensor,
    alpha: Number,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат умножения TT-тензора на скаляр.
    Модифицируем только первое ядро.
    """
    cores = []
    for i, core in enumerate(tt.cores):
        if i == 0:
            cores.append(backend.scale(core, alpha))
        else:
            cores.append(core)  # Передаем по ссылке в целях экономии памяти
            
    return TTTensor(cores)


def tt_hadamard(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат поэлементного произведения (произведения Адамара).
    """
    d = tt1.order
    cores = []
    
    for k in range(d):
        A = tt1.cores[k]
        B = tt2.cores[k]
        rA_prev, n, rA_next = backend.shape(A)
        rB_prev, _, rB_next = backend.shape(B)
        
        # Размеры ядер перемножаются (кронекерово произведение матриц)
        C = backend.zeros((rA_prev * rB_prev, n, rA_next * rB_next))
        
        for iA in range(rA_prev):
            for iB in range(rB_prev):
                iC = iA * rB_prev + iB
                for j in range(n):
                    for mA in range(rA_next):
                        for mB in range(rB_next):
                            mC = mA * rB_next + mB
                            valA = backend.get_element(A, (iA, j, mA))
                            valB = backend.get_element(B, (iB, j, mB))
                            backend.set_element(C, (iC, j, mC), valA * valB)
                            
        cores.append(C)
        
    return TTTensor(cores)


def tt_dot(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> Number:
    """
    Возвращает скалярное произведение двух TT-тензоров: <tt1, tt2>.
    """
    d = tt1.order
    Z = backend.ones((1, 1))  # Начальная матрица Z_0 размера 1x1
    
    for k in range(d):
        A = tt1.cores[k]
        B = tt2.cores[k]
        rA_prev, n, rA_next = backend.shape(A)
        rB_prev, _, rB_next = backend.shape(B)
        
        Z_new = backend.zeros((rA_next, rB_next))
        
        for i in range(n):
            # Последовательная свертка: Z_new += A_i^T @ Z_old @ B_i
            for cA in range(rA_next):
                for cB in range(rB_next):
                    s = 0.0
                    for rA in range(rA_prev):
                        for rB in range(rB_prev):
                            a_val = backend.get_element(A, (rA, i, cA))
                            b_val = backend.get_element(B, (rB, i, cB))
                            z_val = backend.get_element(Z, (rA, rB))
                            s += a_val * z_val * b_val
                            
                    cur = backend.get_element(Z_new, (cA, cB))
                    backend.set_element(Z_new, (cA, cB), cur + s)
                    
        Z = Z_new
        
    return backend.get_element(Z, (0, 0))


def tt_norm(
    tt: TTTensor,
    backend: BackendInterface
) -> float:
    """
    Возвращает Фробениусову норму TT-тензора.
    """
    dot_product = tt_dot(tt, tt, backend)
    # Защита от потери точности при операциях с плавающей точкой
    return math.sqrt(max(0.0, float(dot_product)))


def tt_diff_norm(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> float:
    """
    Возвращает норму разности: ||tt1 - tt2||_F.
    """
    norm1_sq = tt_dot(tt1, tt1, backend)
    norm2_sq = tt_dot(tt2, tt2, backend)
    dot12 = tt_dot(tt1, tt2, backend)
    
    # ||A - B||^2 = ||A||^2 + ||B||^2 - 2<A,B>
    diff_sq = norm1_sq + norm2_sq - 2.0 * dot12
    return math.sqrt(max(0.0, float(diff_sq)))