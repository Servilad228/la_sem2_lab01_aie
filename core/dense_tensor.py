# core/dense_tensor.py

"""Функции для работы с тензорами в стандартной плотной форме."""

from __future__ import annotations

import random
import math

from core.utils import (
    validate_shape,
    compute_size,
    compute_strides,
    multi_index_to_flat,
    flat_to_multi_index,
    check_shapes_match,
)


class DenseTensor:
    """
    Плотный тензор произвольного порядка.
    """

    __slots__ = ('shape', 'ndim', 'size', 'data', 'strides')

    # ────────────────────────────────────────────
    # Конструкторы
    # ────────────────────────────────────────────

    def __init__(
        self,
        shape: tuple[int, ...] | list[int],
        data: list[float] | None = None,
        fill: float = 0.0
    ) -> None:
        self.shape = validate_shape(shape)
        self.ndim = len(self.shape)
        self.size = compute_size(self.shape)
        self.strides = compute_strides(self.shape)
        
        if data is None:
            self.data = [float(fill)] * self.size
        else:
            if len(data) != self.size:
                raise ValueError(f"Размер данных ({len(data)}) не совпадает с заявленной формой (ожидалось {self.size})")
            self.data = list(data)

    @staticmethod
    def zeros(shape: tuple[int, ...] | list[int]) -> DenseTensor:
        return DenseTensor(shape, fill=0.0)

    @staticmethod
    def ones(shape: tuple[int, ...] | list[int]) -> DenseTensor:
        return DenseTensor(shape, fill=1.0)

    @staticmethod
    def random(
        shape: tuple[int, ...] | list[int],
        low: int = -5,
        high: int = 5,
        integer: bool = True,
        seed: int | None = None
    ) -> DenseTensor:
        if seed is not None:
            random.seed(seed)
            
        size = compute_size(validate_shape(shape))
        if integer:
            data = [float(random.randint(int(low), int(high))) for _ in range(size)]
        else:
            data = [random.uniform(low, high) for _ in range(size)]
            
        return DenseTensor(shape, data=data)

    @staticmethod
    def from_nested_list(nested: list | tuple) -> DenseTensor:
        shape = []
        curr = nested
        # Добавили проверку на tuple
        while isinstance(curr, (list, tuple)):
            shape.append(len(curr))
            if len(curr) > 0:
                curr = curr[0]
            else:
                break
                
        data = []
        def flatten(lst, depth):
            if depth == len(shape):
                data.append(float(lst))
                return
            for item in lst:
                flatten(item, depth + 1)
                
        flatten(nested, 0)
        return DenseTensor(shape, data=data)

    # ────────────────────────────────────────────
    # Индексация
    # ────────────────────────────────────────────

    def _validate_index(
        self,
        multi_index: tuple[int, ...] | int
    ) -> tuple[int, ...]:
        if isinstance(multi_index, int):
            if self.ndim == 1:
                return (multi_index,)
            else:
                return flat_to_multi_index(multi_index, self.shape)
                
        if isinstance(multi_index, list):
            multi_index = tuple(multi_index)
            
        if len(multi_index) != self.ndim:
            raise IndexError(f"Неверное число индексов: ожидалось {self.ndim}, получено {len(multi_index)}")
            
        return multi_index

    def __getitem__(self, multi_index: tuple[int, ...] | int) -> float:
        norm_idx = self._validate_index(multi_index)
        flat_idx = multi_index_to_flat(norm_idx, self.strides)
        return self.data[flat_idx]

    def __setitem__(
        self,
        multi_index: tuple[int, ...] | int,
        value: float
    ) -> None:
        norm_idx = self._validate_index(multi_index)
        flat_idx = multi_index_to_flat(norm_idx, self.strides)
        self.data[flat_idx] = float(value)

    # ────────────────────────────────────────────
    # Преобразования формы
    # ────────────────────────────────────────────

    def reshape(self, new_shape: tuple[int, ...] | list[int]) -> DenseTensor:
        new_shape = validate_shape(new_shape)
        if compute_size(new_shape) != self.size:
            raise ValueError("Новая форма несовместима с количеством элементов тензора")
        return DenseTensor(new_shape, data=self.data[:])

    def unfolding(self, mode: int) -> DenseTensor:
        if not (0 <= mode < self.ndim):
            raise ValueError(f"Некорректная мода: {mode}")
            
        n_row = self.shape[mode]
        n_col = self.size // n_row
        result = DenseTensor((n_row, n_col), fill=0.0)
        
        col_shape = tuple(self.shape[i] for i in range(self.ndim) if i != mode)
        col_strides = compute_strides(col_shape)
        
        for flat_idx in range(self.size):
            multi_idx = flat_to_multi_index(flat_idx, self.shape)
            row = multi_idx[mode]
            
            col_multi_idx = tuple(multi_idx[i] for i in range(self.ndim) if i != mode)
            col = multi_index_to_flat(col_multi_idx, col_strides)
            
            result.data[row * n_col + col] = self.data[flat_idx]
            
        return result

    def left_unfolding(self, k: int) -> DenseTensor:
        if not (0 <= k < self.ndim - 1):
            raise ValueError(f"Некорректная граница разбиения: {k}")
            
        n_row = 1
        for i in range(k + 1):
            n_row *= self.shape[i]
            
        n_col = self.size // n_row
        
        # Левая развертка эквивалентна изменению формы в C-order
        return self.reshape((n_row, n_col))

    # ────────────────────────────────────────────
    # Копирование
    # ────────────────────────────────────────────

    def copy(self) -> DenseTensor:
        return DenseTensor(self.shape, data=self.data[:])

    # ────────────────────────────────────────────
    # Арифметика
    # ────────────────────────────────────────────

    def norm(self) -> float:
        return math.sqrt(sum(x * x for x in self.data))

    def __add__(self, other: DenseTensor) -> DenseTensor:
        check_shapes_match(self.shape, other.shape)
        new_data = [a + b for a, b in zip(self.data, other.data)]
        return DenseTensor(self.shape, data=new_data)

    def __sub__(self, other: DenseTensor) -> DenseTensor:
        check_shapes_match(self.shape, other.shape)
        new_data = [a - b for a, b in zip(self.data, other.data)]
        return DenseTensor(self.shape, data=new_data)

    def __mul__(self, scalar: float | int) -> DenseTensor:
        new_data = [x * scalar for x in self.data]
        return DenseTensor(self.shape, data=new_data)

    def __rmul__(self, scalar: float | int) -> DenseTensor:
        return self.__mul__(scalar)

    def __neg__(self) -> DenseTensor:
        new_data = [-x for x in self.data]
        return DenseTensor(self.shape, data=new_data)

    # ────────────────────────────────────────────
    # Сравнение и отладка
    # ────────────────────────────────────────────

    def allclose(
        self,
        other: DenseTensor,
        atol: float = 1e-8,
        rtol: float = 1e-5
    ) -> bool:
        if self.shape != other.shape:
            return False
            
        for a, b in zip(self.data, other.data):
            if abs(a - b) > atol + rtol * max(abs(a), abs(b)):
                return False
        return True

    def to_nested_list(self) -> list:
        def build_nested(offset, current_dim):
            if current_dim == self.ndim - 1:
                return self.data[offset : offset + self.shape[current_dim]]
            
            stride = self.strides[current_dim]
            return [build_nested(offset + i * stride, current_dim + 1) 
                    for i in range(self.shape[current_dim])]
            
        if self.ndim == 0:
            return []
        return build_nested(0, 0)

    def __repr__(self) -> str:
        return f"DenseTensor(shape={self.shape}, size={self.size})"

    def __str__(self) -> str:
        return self.__repr__()