"""
Testing utils for ML
"""

from typing import Iterable, Callable, Union, Sized
from sklearn.model_selection import GroupShuffleSplit
from i2.signatures import call_forgivingly


def keys_aligned_list(iterable_spec, keys):
    """

    :param iterable_spec:
    :param keys:
    :return:
    """
    if iterable_spec is None:
        return None
    elif isinstance(iterable_spec, Callable):
        return list(map(iterable_spec, keys))
    elif isinstance(iterable_spec, Iterable):
        iterable_spec = list(iterable_spec)
        assert len(iterable_spec) == len(keys)
        return iterable_spec


def train_test_split_keys(
    keys: Iterable,
    key_to_tag: Union[Callable, Iterable, None] = None,
    key_to_group: Union[Callable, Iterable, None] = None,
    test_size=None,
    train_size=None,
    random_state=None,
    n_splits=1,
):
    """Split keys into train and test lists.

    :param keys: keys to be split
    :param key_to_tag: keys-aligned iterable of tags (a.k.a y/classes in
    sklearn speak) or function to compute these from keys
    :param key_to_group: keys-aligned iterable of groups or function to compute
    these from keys

    >>> keys = range(100)
    >>> def mod5(x):
    ...     return x % 5
    >>> train_idx, test_idx = train_test_split_keys(keys, key_to_group=mod5,
    ...     train_size=.5, random_state=42)

    Observe here that though `train_size=.5`, the proportion is not 50/50.
    That's because the group constraint, imposed by the key_to_group argument
    produces only 5 groups.

    >>> len(train_idx), len(test_idx)
    (40, 60)

    But especially, see that though there's a lot of train and test indices,
    within train, there's only 2 unique groups (all 0 or 3 modulo 5)
    and only 3 unique groups (1, 2, 4 modulo 5) within test indices.

    >>> assert set(map(mod5, train_idx)) == {0, 3}
    >>> assert set(map(mod5, test_idx)) == {1, 2, 4}

    """
    splitter = call_forgivingly(
        GroupShuffleSplit, **locals()
    )  # calls GroupShuffleSplit on relevant inputs

    X = list(keys)
    y = keys_aligned_list(key_to_tag, keys)
    groups = keys_aligned_list(key_to_group, keys)

    n = splitter.get_n_splits(X, y, groups)
    if n == 1:
        return next(splitter.split(X, y, groups))
    else:
        return splitter.split(X, y, groups)
