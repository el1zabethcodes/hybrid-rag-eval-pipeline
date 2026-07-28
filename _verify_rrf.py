import importlib.util, sys

# Load rrf module directly to bypass the package __init__ that imports
# dense_embedder (which needs sentence_transformers, not yet installed)
spec = importlib.util.spec_from_file_location(
    "rrf",
    r"src\retrieval\rrf.py",
)
rrf_mod = importlib.util.load_from_spec = None  # not needed

# Simpler: just patch sys.path and import submodule directly
import sys
import importlib

# Temporarily replace __init__ import side-effect by importing submodule directly
import types
pkg = types.ModuleType("src.retrieval")
sys.modules.setdefault("src.retrieval", pkg)

from src.retrieval.rrf import rrf_merge
from src.models import Chunk, ScoredChunk

def make(cid):
    return ScoredChunk(chunk=Chunk(id=cid, document_id="doc", text="t", chunk_index=0), score=0.0)

dense = [make("a"), make("b"), make("c")]
sparse = [make("b"), make("d"), make("a")]
r = rrf_merge(dense, sparse, rrf_k=60, top_k=4)
ids = [x.chunk.id for x in r]
scores = [round(x.score, 6) for x in r]
print("IDs:", ids)
print("Scores:", scores)

# Verify expected ranking:
# b: 1/(60+1) + 1/(60+1) = 2/61 ≈ 0.032787  (rank 1 in both)
# a: 1/(60+1) + 1/(60+3) = 1/61 + 1/63 ≈ 0.032307  (rank 1 dense, rank 3 sparse)
# c: 1/(60+3) = 1/63 ≈ 0.015873  (rank 3 dense only)
# d: 1/(60+2) = 1/62 ≈ 0.016129  (rank 2 sparse only)
assert ids == ["b", "a", "d", "c"], f"Unexpected order: {ids}"

# Verify empty inputs
empty = rrf_merge([], [], rrf_k=60, top_k=10)
assert empty == [], f"Expected empty, got {empty}"

# Verify top_k truncation
r2 = rrf_merge(dense, sparse, rrf_k=60, top_k=2)
assert len(r2) == 2, f"Expected 2, got {len(r2)}"

# Verify single list (sparse empty)
r3 = rrf_merge(dense, [], rrf_k=60, top_k=10)
assert len(r3) == 3
assert [x.chunk.id for x in r3] == ["a", "b", "c"], f"Unexpected: {[x.chunk.id for x in r3]}"

# Verify no-overlap lists
r4 = rrf_merge([make("x"), make("y")], [make("z")], rrf_k=60, top_k=10)
ids4 = [x.chunk.id for x in r4]
assert set(ids4) == {"x", "y", "z"}, f"Missing chunks: {ids4}"

print("All assertions passed.")
