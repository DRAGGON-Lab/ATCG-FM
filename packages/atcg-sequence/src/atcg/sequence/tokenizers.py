"""Small, explicit tokenizers for character- and byte-level experiments."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """Contract used by datasets and inference without a framework dependency."""

    @property
    def vocab_size(self) -> int: ...

    @property
    def pad_id(self) -> int: ...

    @property
    def bos_id(self) -> int: ...

    @property
    def eos_id(self) -> int: ...

    def encode(
        self,
        sequence: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]: ...

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special: bool = True,
        stop_at_eos: bool = True,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class ByteTokenizer:
    """Identity tokenizer over bytes with four reserved special tokens."""

    normalize_uppercase: bool = True

    @property
    def pad_id(self) -> int:
        return 256

    @property
    def bos_id(self) -> int:
        return 257

    @property
    def eos_id(self) -> int:
        return 258

    @property
    def unk_id(self) -> int:
        return 259

    @property
    def vocab_size(self) -> int:
        return 260

    def encode(
        self,
        sequence: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        normalized = sequence.upper() if self.normalize_uppercase else sequence
        try:
            token_ids = list(normalized.encode("ascii"))
        except UnicodeEncodeError as error:
            raise ValueError("byte tokenizer accepts ASCII sequence text only") from error

        if add_bos:
            token_ids.insert(0, self.bos_id)
        if add_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special: bool = True,
        stop_at_eos: bool = True,
    ) -> str:
        byte_values: list[int] = []
        for token_id in token_ids:
            self._validate_id(token_id)
            if token_id == self.eos_id and stop_at_eos:
                break
            if token_id >= 256:
                if skip_special:
                    continue
                raise ValueError(f"cannot decode special token {token_id} as sequence text")
            byte_values.append(token_id)

        try:
            return bytes(byte_values).decode("ascii")
        except UnicodeDecodeError as error:
            raise ValueError("token ids do not decode to ASCII sequence text") from error

    def _validate_id(self, token_id: int) -> None:
        if not 0 <= token_id < self.vocab_size:
            raise ValueError(f"token id {token_id} is outside vocabulary")


DEFAULT_IUPAC_DNA_ALPHABET = "ACGTRYSWKMBDHVN-"


@dataclass(frozen=True, slots=True)
class FixedAlphabetTokenizer:
    """Character tokenizer for controlled nucleotide-alphabet ablations."""

    alphabet: str = DEFAULT_IUPAC_DNA_ALPHABET
    normalize_uppercase: bool = True
    strict: bool = True
    _token_by_character: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        normalized = self.alphabet.upper() if self.normalize_uppercase else self.alphabet
        if not normalized:
            raise ValueError("alphabet must not be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("alphabet characters must be unique")
        if any(character.isspace() for character in normalized):
            raise ValueError("alphabet must not contain whitespace")

        object.__setattr__(self, "alphabet", normalized)
        object.__setattr__(
            self,
            "_token_by_character",
            {character: index for index, character in enumerate(normalized)},
        )

    @property
    def pad_id(self) -> int:
        return len(self.alphabet)

    @property
    def bos_id(self) -> int:
        return len(self.alphabet) + 1

    @property
    def eos_id(self) -> int:
        return len(self.alphabet) + 2

    @property
    def unk_id(self) -> int:
        return len(self.alphabet) + 3

    @property
    def vocab_size(self) -> int:
        return len(self.alphabet) + 4

    def encode(
        self,
        sequence: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        normalized = sequence.upper() if self.normalize_uppercase else sequence
        token_ids: list[int] = []
        for offset, character in enumerate(normalized):
            token_id = self._token_by_character.get(character)
            if token_id is None:
                if self.strict:
                    raise ValueError(
                        f"character {character!r} at offset {offset} is not in alphabet"
                    )
                token_id = self.unk_id
            token_ids.append(token_id)

        if add_bos:
            token_ids.insert(0, self.bos_id)
        if add_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special: bool = True,
        stop_at_eos: bool = True,
    ) -> str:
        characters: list[str] = []
        for token_id in token_ids:
            if not 0 <= token_id < self.vocab_size:
                raise ValueError(f"token id {token_id} is outside vocabulary")
            if token_id == self.eos_id and stop_at_eos:
                break
            if token_id < len(self.alphabet):
                characters.append(self.alphabet[token_id])
            elif token_id == self.unk_id:
                characters.append("?")
            elif not skip_special:
                raise ValueError(f"cannot decode special token {token_id} as sequence text")
        return "".join(characters)
