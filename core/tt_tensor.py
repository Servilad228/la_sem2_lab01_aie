# core/tt_tensor.py

"""
Тензор в TT-формате (Tensor Train).

TT-тензор порядка d с shape (n_0, n_1, ..., n_{d-1}) хранится как
список d ядер (cores), где k-е ядро — это 3D DenseTensor с shape:
    (r_k, n_k, r_{k+1})

Граничные условия: r_0 = r_d = 1.

TT-ранги: (r_0, r_1, ..., r_d) = (1, r_1, ..., r_{d-1}, 1).
"""

from __future__ import annotations

import random

from core.dense_tensor import DenseTensor
from core.utils import validate_shape, compute_size, flat_to_multi_index


class TTTensor:
    """
    Тензор в TT-формате.
    """

    __slots__ = ('cores', 'order', 'shape', 'ranks')

    # ────────────────────────────────────────────
    # Конструкторы
    # ────────────────────────────────────────────

    def __init__(self, cores: list[DenseTensor]) -> None:
        """
        Создаёт TT-тензор из списка ядер.
        """
        if not cores:
            raise ValueError("Список ядер не может быть пустым.")
            
        self.cores = cores
        self.order = len(cores)
        
        shape_list = []
        ranks_list = [cores[0].shape[0]]
        
        for k in range(self.order):
            core = cores[k]
            if core.ndim != 3:
                raise ValueError(f"Ядро {k} должно быть 3D-тензором, получено {core.ndim}D.")
                
            r_prev, n_k, r_k = core.shape
            
            if ranks_list[-1] != r_prev:
                raise ValueError(f"Несовпадение рангов: ядро {k} ожидает входящий ранг {ranks_list[-1]}, а имеет {r_prev}.")
                
            shape_list.append(n_k)
            ranks_list.append(r_k)
            
        if ranks_list[0] != 1 or ranks_list[-1] != 1:
            raise ValueError(f"Граничные ранги должны быть равны 1. Получено: r_0={ranks_list[0]}, r_d={ranks_list[-1]}.")
            
        self.shape = tuple(shape_list)
        self.ranks = tuple(ranks_list)

    @staticmethod
    def random(shape, ranks, seed=None):
        """
        Создаёт случайный TT-тензор с заданными рангами.
        """
        if seed is not None:
            random.seed(seed)
            
        shape = validate_shape(shape)
        d = len(shape)
        
        if isinstance(ranks, (tuple, list)):
            if len(ranks) == d - 1:
                ranks_full = [1] + list(ranks) + [1]
            elif len(ranks) == d + 1:
                ranks_full = list(ranks)
            else:
                raise ValueError("Неверная длина массива рангов.")
        else:
            raise TypeError("Ranks должен быть списком или кортежем.")
            
        cores = []
        for k in range(d):
            r_prev = ranks_full[k]
            n_k = shape[k]
            r_k = ranks_full[k+1]
            core = DenseTensor.random((r_prev, n_k, r_k), seed=seed)
            cores.append(core)
            
        return TTTensor(cores)

    # ────────────────────────────────────────────
    # Доступ к элементам
    # ────────────────────────────────────────────

    def get_element(
        self,
        indices: tuple[int, ...] | list[int]
    ) -> float:
        """
        Возвращает элемент TT-тензора по его мультииндексу.
        """
        if len(indices) != self.order:
            raise ValueError(f"Ожидалось {self.order} индексов, получено {len(indices)}.")
            
        # Начинаем с первого ядра. Срез G_1[i_1] имеет размер (1, r_1).
        # Представляем его как вектор длины r_1.
        current_vector = [self.cores[0][0, indices[0], r] for r in range(self.ranks[1])]
        
        # Последовательно умножаем вектор на срезы (матрицы) следующих ядер
        for k in range(1, self.order):
            next_core = self.cores[k]
            i_k = indices[k]
            r_prev = self.ranks[k]
            r_next = self.ranks[k+1]
            
            next_vector = [0.0] * r_next
            for c in range(r_next):
                val = 0.0
                for r in range(r_prev):
                    val += current_vector[r] * next_core[r, i_k, c]
                next_vector[c] = val
                
            current_vector = next_vector
            
        # На последнем шаге вектор имеет длину r_d = 1
        return current_vector[0]

    # ────────────────────────────────────────────
    # Восстановление полного тензора
    # ────────────────────────────────────────────

    def full(self) -> DenseTensor:
        """Возвращает полный DenseTensor из его TT-формата."""
        total_elements = compute_size(self.shape)
        data = [0.0] * total_elements
        
        for flat_idx in range(total_elements):
            multi_idx = flat_to_multi_index(flat_idx, self.shape)
            data[flat_idx] = self.get_element(multi_idx)
            
        return DenseTensor(self.shape, data=data)

    # ────────────────────────────────────────────
    # Информация и отладка
    # ────────────────────────────────────────────

    def core_sizes(self) -> list[tuple[int, ...]]:
        """Возвращает размеры всех ядер."""
        return [core.shape for core in self.cores]

    def total_storage(self) -> int:
        """Возвращает общее число элементов во всех ядрах."""
        return sum(core.size for core in self.cores)

    def compression_ratio(self) -> float:
        """Возвращает отношение числа элементов полного тензора к числу элементов TT-тензора."""
        full_size = compute_size(self.shape)
        storage = self.total_storage()
        return full_size / storage if storage > 0 else 0.0

    def copy(self) -> TTTensor:
        """Возвращает глубокую копию TT-тензора."""
        return TTTensor([core.copy() for core in self.cores])

    def __repr__(self) -> str:
        """Возвращает строковое представление TT-тензора для отладки."""
        info = (
            f"TTTensor:\n"
            f"  - Order: {self.order}\n"
            f"  - Shape: {self.shape}\n"
            f"  - TT-ranks: {self.ranks}\n"
            f"  - Cores shapes: {self.core_sizes()}\n"
            f"  - Storage size: {self.total_storage()} elements\n"
            f"  - Compression: {self.compression_ratio():.2f}x"
        )
        return info

    def __str__(self) -> str:
        """Возвращает строковое представление TT-тензора."""
        return self.__repr__()