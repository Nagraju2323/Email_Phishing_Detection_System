"""Find near-duplicate email groups and make a reproducible grouped split.

Similarity is based on sets of word trigrams. A deterministic CRC32 SimHash
acts as a fast candidate filter; candidates are verified with Jaccard
similarity before they are joined into a group.
"""
import re
import zlib

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


SIMILARITY_THRESHOLD = 0.90
SIMHASH_MAX_DISTANCE = 3
RANDOM_STATE = 42


def _shingle_hashes(text):
    """Return sorted unique 32-bit hashes for all word trigrams in text."""
    words = re.findall(r"\b[\w@.+:/-]+\b", text)
    if len(words) >= 3:
        values = (
            zlib.crc32((words[i] + "\x1f" + words[i + 1] + "\x1f" + words[i + 2]).encode("utf-8"))
            for i in range(len(words) - 2)
        )
    else:
        values = iter((zlib.crc32("\x1f".join(words).encode("utf-8")),))
    return np.unique(np.fromiter(values, dtype=np.uint32))


def _simhash(shingles):
    """Build a deterministic 32-bit SimHash signature from unique shingles."""
    signature = 0
    for bit in range(32):
        ones = np.count_nonzero((shingles >> np.uint32(bit)) & np.uint32(1))
        if ones * 2 >= len(shingles):
            signature |= 1 << bit
    return signature


def build_similarity_groups(texts, labels=None, prior_split=None):
    """Return connected-component IDs and leakage/group statistics.

    Two messages are linked when their 32-bit signatures are within three
    bits and their word-trigram Jaccard similarity is at least 0.90. Four
    8-bit LSH bands ensure pairs within that Hamming distance share a bucket.
    Connected components keep transitive template variants together.

    `prior_split`, when supplied, is a sequence of 0/1 assignments used only
    to measure cross-split near-duplicate overlap in an earlier split.
    """
    count = len(texts)
    shingles = []
    signatures = []
    for text in texts:
        current = _shingle_hashes(str(text))
        shingles.append(current)
        signatures.append(_simhash(current))

    parent = list(range(count))
    sizes = [1] * count

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if sizes[left_root] < sizes[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        sizes[left_root] += sizes[right_root]

    buckets = [{} for _ in range(4)]
    qualifying_pairs = 0
    cross_split_pairs = 0
    cross_split_test_rows = set()
    label_conflict_pairs = 0

    for index, signature in enumerate(signatures):
        candidates = set()
        for band in range(4):
            key = (signature >> (band * 8)) & 255
            candidates.update(buckets[band].get(key, ()))

        current_shingles = shingles[index]
        for other in candidates:
            distance = (signature ^ signatures[other]).bit_count()
            if distance > SIMHASH_MAX_DISTANCE:
                continue
            intersection = np.intersect1d(
                current_shingles, shingles[other], assume_unique=True
            ).size
            union_size = len(current_shingles) + len(shingles[other]) - intersection
            similarity = intersection / union_size if union_size else 1.0
            if similarity < SIMILARITY_THRESHOLD:
                continue

            qualifying_pairs += 1
            union(index, other)
            if labels is not None and labels[index] != labels[other]:
                label_conflict_pairs += 1
            if prior_split is not None and prior_split[index] != prior_split[other]:
                cross_split_pairs += 1
                if prior_split[index] == 1:
                    cross_split_test_rows.add(index)
                else:
                    cross_split_test_rows.add(other)

        for band in range(4):
            key = (signature >> (band * 8)) & 255
            buckets[band].setdefault(key, []).append(index)

    group_by_root = {}
    for index in range(count):
        group_by_root.setdefault(find(index), []).append(index)
    # Assign stable group IDs in order of each component's first row.
    ordered_groups = sorted(group_by_root.values(), key=lambda members: members[0])
    group_ids = np.empty(count, dtype=np.int64)
    for group_id, members in enumerate(ordered_groups):
        group_ids[members] = group_id

    group_sizes = np.fromiter((len(members) for members in ordered_groups), dtype=np.int64)
    stats = {
        "group_count": len(ordered_groups),
        "singleton_groups": int(np.count_nonzero(group_sizes == 1)),
        "largest_group": int(group_sizes.max()) if len(group_sizes) else 0,
        "qualifying_pairs": qualifying_pairs,
        "cross_split_pairs": cross_split_pairs,
        "cross_split_test_rows": len(cross_split_test_rows),
        "label_conflict_pairs": label_conflict_pairs,
        "groups": ordered_groups,
    }
    return group_ids, stats


def choose_stratified_group_split(labels, groups, test_size=0.20, random_state=RANDOM_STATE):
    """Select the 5-fold stratified group fold closest to target size and mix."""
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=random_state
    )
    overall_rates = np.bincount(labels.astype(int), minlength=2) / len(labels)
    candidates = []
    for train_indices, test_indices in splitter.split(np.zeros(len(labels)), labels, groups):
        test_rates = np.bincount(labels[test_indices].astype(int), minlength=2) / len(test_indices)
        size_error = abs(len(test_indices) / len(labels) - test_size)
        class_error = np.abs(test_rates - overall_rates).sum()
        candidates.append((size_error + class_error, train_indices, test_indices))
    _, train_indices, test_indices = min(candidates, key=lambda item: item[0])

    if set(groups[train_indices]).intersection(groups[test_indices]):
        raise RuntimeError("Group leakage detected: a group appears in both splits.")
    return train_indices, test_indices
