# core/utils.py

"""Вспомогательные функции для работы с тензорами."""

def validate_shape(shape: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """
    Проверяет корректность формы тензора и приводит её к стандартному виду.
    """
    if not isinstance(shape, (tuple, list)):
        raise TypeError("Shape должен быть объектом tuple или list")
    
    normalized_shape = tuple(shape)
    for dim in normalized_shape:
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"Все размерности должны быть положительными целыми числами, получено: {shape}")
            
    return normalized_shape


def compute_size(shape: tuple[int, ...]) -> int:
    """
    Возвращает общее число элементов тензора заданной формы.
    """
    if not shape:
        return 0
    size = 1
    for dim in shape:
        size *= dim
    return size


def compute_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """
    Возвращает кортеж strides, содержащий для каждой моды k свой strides[k].
    
    Для C-order шаг по последней моде всегда равен 1.
    """
    if not shape:
        return ()
    
    strides = [1] * len(shape)
    # Идем с предпоследнего элемента до нулевого
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
        
    return tuple(strides)


def multi_index_to_flat(multi_index: tuple[int, ...], strides: tuple[int, ...]) -> int:
    """
    Возвращает позицию элемента в плоском списке данных по его
    многомерным координатам и заранее вычисленным strides.
    """
    flat_index = 0
    for i, s in zip(multi_index, strides):
        flat_index += i * s
    return flat_index


def flat_to_multi_index(flat_index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    """
    Возвращает мультииндекс на основе плоского индекса.
    """
    multi_index = [0] * len(shape)
    current_idx = flat_index
    
    # Идем с конца (от последней моды к первой), беря остаток от деления
    for i in range(len(shape) - 1, -1, -1):
        multi_index[i] = current_idx % shape[i]
        current_idx //= shape[i]
        
    return tuple(multi_index)


def check_shapes_match(shape1: tuple[int, ...], shape2: tuple[int, ...]) -> None:
    """
    Проверяет совпадение форм двух тензоров.
    """
    if shape1 != shape2:
        raise ValueError(f"Формы тензоров не совпадают: {shape1} != {shape2}")