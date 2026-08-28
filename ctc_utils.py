"""Strict CTC label conversion for the revised CRNN experiments."""

import torch


class CTCLabelConverter:
    """Map charset characters to indices, with index zero reserved for CTC blank."""

    def __init__(self, charset):
        if len(charset) != len(set(charset)):
            raise ValueError("Charset contains duplicate characters")
        self.charset = charset
        self.blank_idx = 0
        self.char_to_idx = {character: index + 1 for index, character in enumerate(charset)}
        self.idx_to_char = {index + 1: character for index, character in enumerate(charset)}

    @property
    def num_classes(self):
        return len(self.charset) + 1

    def encode_batch(self, labels):
        encoded = []
        lengths = []
        for label in labels:
            unknown = sorted(set(label) - set(self.charset))
            if unknown:
                raise ValueError(f"Unknown characters in label {label!r}: {unknown}")
            values = [self.char_to_idx[character] for character in label]
            encoded.extend(values)
            lengths.append(len(values))
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)

    def decode_indices(self, indices):
        if torch.is_tensor(indices):
            indices = indices.tolist()
        result = []
        previous = None
        for index in indices:
            index = int(index)
            if index != self.blank_idx and index != previous:
                result.append(self.idx_to_char[index])
            previous = index
        return "".join(result)

    def decode_logits(self, logits):
        """Decode logits with explicit (batch, timestep, class) layout."""

        if logits.ndim != 3:
            raise ValueError(f"Expected B x T x C logits, got {tuple(logits.shape)}")
        indices = logits.argmax(dim=2)
        return [self.decode_indices(row) for row in indices]


def minimum_ctc_timesteps(label):
    """Return label length plus blanks required between adjacent repeated characters."""

    adjacent_repeats = sum(left == right for left, right in zip(label, label[1:]))
    return len(label) + adjacent_repeats
