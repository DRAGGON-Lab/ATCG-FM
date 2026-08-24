"""Small Newick utilities for deterministic GTDB neighborhood selection."""

from __future__ import annotations

from dataclasses import dataclass, field


def normalize_gtdb_accession(value: str) -> str:
    """Remove the GTDB source prefix while retaining the assembly version."""

    return value.removeprefix("RS_").removeprefix("GB_")


@dataclass(slots=True)
class _Node:
    name: str = ""
    length: float = 0.0
    children: list[_Node] = field(default_factory=lambda: list[_Node]())
    parent: _Node | None = field(default=None, repr=False)


class _NewickParser:
    def __init__(self, text: str) -> None:
        self.tokens = _tokens(text)
        self.index = 0

    def parse(self) -> _Node:
        root = self._subtree()
        if self._peek() == ";":
            self.index += 1
        if self.index != len(self.tokens):
            raise ValueError("unexpected trailing Newick tokens")
        return root

    def _subtree(self) -> _Node:
        children: list[_Node] = []
        if self._peek() == "(":
            self.index += 1
            while True:
                children.append(self._subtree())
                token = self._take()
                if token == ")":
                    break
                if token != ",":
                    raise ValueError("expected ',' or ')' in Newick tree")
        name = ""
        if self._peek() not in {None, ":", ",", ")", ";"}:
            name = self._take()
        length = 0.0
        if self._peek() == ":":
            self.index += 1
            try:
                length = float(self._take())
            except ValueError as error:
                raise ValueError("invalid Newick branch length") from error
        node = _Node(name=name, length=length, children=children)
        for child in children:
            child.parent = node
        return node

    def _peek(self) -> str | None:
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def _take(self) -> str:
        value = self._peek()
        if value is None:
            raise ValueError("unexpected end of Newick tree")
        self.index += 1
        return value


def _tokens(text: str) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(text):
        character = text[index]
        if character.isspace():
            index += 1
        elif character in "(),:;":
            result.append(character)
            index += 1
        elif character in {"'", '"'}:
            quote = character
            index += 1
            start = index
            while index < len(text) and text[index] != quote:
                index += 1
            if index == len(text):
                raise ValueError("unterminated quoted Newick label")
            result.append(text[start:index])
            index += 1
        else:
            start = index
            while index < len(text) and not text[index].isspace() and text[index] not in "(),:;":
                index += 1
            result.append(text[start:index])
    return result


def nearest_accessions(
    newick: str,
    *,
    anchor_accession: str,
    candidate_accessions: set[str],
    limit: int,
) -> tuple[tuple[str, float], ...]:
    """Return candidate leaves in ascending patristic distance from an anchor leaf."""

    if limit < 1:
        raise ValueError("neighbor limit must be positive")
    root = _NewickParser(newick).parse()
    leaves = _leaves(root)
    by_accession = {normalize_gtdb_accession(node.name): node for node in leaves}
    anchor_key = normalize_gtdb_accession(anchor_accession)
    try:
        anchor = by_accession[anchor_key]
    except KeyError as error:
        message = f"anchor accession {anchor_accession!r} is absent from the tree"
        raise ValueError(message) from error
    anchor_distances = _ancestor_distances(anchor)
    rows: list[tuple[str, float]] = []
    for accession in sorted(candidate_accessions):
        key = normalize_gtdb_accession(accession)
        node = by_accession.get(key)
        if node is None:
            continue
        distance = _distance(anchor, node, anchor_distances)
        rows.append((key, distance))
    rows.sort(key=lambda row: (row[1], row[0]))
    return tuple(rows[:limit])


def _leaves(node: _Node) -> list[_Node]:
    if not node.children:
        return [node]
    return [leaf for child in node.children for leaf in _leaves(child)]


def _ancestor_distances(node: _Node) -> dict[int, float]:
    result: dict[int, float] = {id(node): 0.0}
    distance = 0.0
    current = node
    while current.parent is not None:
        distance += current.length
        current = current.parent
        result[id(current)] = distance
    return result


def _distance(node: _Node, other: _Node, first: dict[int, float]) -> float:
    distance = 0.0
    current = other
    while id(current) not in first:
        if current.parent is None:
            raise ValueError("Newick leaves do not share a root")
        distance += current.length
        current = current.parent
    return first[id(current)] + distance
