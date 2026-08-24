"""Read-only access to the shared ATCG-FM data store."""

from atcg.data.phylogeny import nearest_accessions, normalize_gtdb_accession
from atcg.data.store import DEFAULT_BUCKET, DataStore

__all__ = ["DEFAULT_BUCKET", "DataStore", "nearest_accessions", "normalize_gtdb_accession"]
