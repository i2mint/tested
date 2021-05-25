"""
Testing utils for ML
"""


from typing import Iterable, Optional, Callable
from sklearn.model_selection import GroupShuffleSplit
from i2.signatures import call_forgivingly


def train_test_split_keys(
    keys: Iterable,
    key_to_tag: Optional[Callable] = None,
    key_to_group: Optional[Callable] = None,
    test_size=None,
    train_size=None,
    random_state=None,
    n_splits=1,
):
    """

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

    y = None
    if key_to_tag is not None:
        y = list(map(key_to_tag, keys))

    groups = None
    if key_to_group is not None:
        groups = list(map(key_to_group, keys))

    n = splitter.get_n_splits(X, y, groups)
    if n == 1:
        return next(splitter.split(X, y, groups))
    else:
        return splitter.split(X, y, groups)
