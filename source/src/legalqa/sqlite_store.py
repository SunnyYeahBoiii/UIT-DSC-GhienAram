from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .chunking import chunk_documents
from .data import Chunk, iter_corpus, load_qa, write_jsonl
from .legal_graph import RelationEdge, document_aliases, extract_relation_mentions
from .text import citations, document_numbers, legal_tokens, numbers, overlap_ratio, words


STORE_SCHEMA_VERSION = "5"


@dataclass(frozen=True)
class SQLiteHit:
    rowid: int
    chunk: Chunk
    lexical_score: float


def corpus_fingerprint(path: str | Path) -> str:
    source = Path(path)
    if source.is_dir():
        files = sorted(source.glob("*.json"))
    else:
        files = [source]
    digest = hashlib.sha256()
    for item in files:
        stat = item.stat()
        digest.update(str(item.name).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def _quote_fts(token: str) -> str:
    return f'"{token.replace(chr(34), chr(34) * 2)}"'


def _fts_queries(terms: list[str], max_terms: int = 14) -> list[str]:
    terms = terms[:max_terms]
    if not terms:
        return []
    quoted = [_quote_fts(token) for token in terms]
    queries = []
    if len(quoted) >= 5:
        queries.append(" AND ".join(quoted[:5]))
    if len(quoted) >= 3:
        queries.append(" AND ".join(quoted[:3]))
    queries.append(" OR ".join(quoted[:10]))
    return list(dict.fromkeys(queries))


class SQLiteLegalStore:
    def __init__(self, path: str | Path, readonly: bool = False) -> None:
        self.path = Path(path)
        if readonly:
            uri = f"file:{self.path.as_posix()}?mode=ro"
            self.connection = sqlite3.connect(uri, uri=True)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self._feature_cache: OrderedDict[int, tuple[set[str], set[str], set[str]]] = OrderedDict()
        self._alignment_cache: OrderedDict[int, tuple[set[str], set[str], set[str]]] = OrderedDict()
        self._has_vocab = False
        try:
            self.connection.execute(
                "CREATE VIRTUAL TABLE temp.legalqa_vocab USING fts5vocab(main, documents_fts, 'row')"
            )
            self._has_vocab = True
        except sqlite3.OperationalError:
            # The table is unavailable while the database is initially being built.
            self._has_vocab = False

    def close(self) -> None:
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def manifest(self) -> dict[str, str]:
        try:
            rows = self.connection.execute("SELECT key, value FROM manifest").fetchall()
        except sqlite3.OperationalError:
            return {}
        return {row["key"]: row["value"] for row in rows}

    def features_for_hit(self, hit: SQLiteHit) -> tuple[set[str], set[str], set[str]]:
        cached = self._feature_cache.get(hit.rowid)
        if cached is not None:
            self._feature_cache.move_to_end(hit.rowid)
            return cached
        value = (set(legal_tokens(hit.chunk.text)), citations(hit.chunk.text), numbers(hit.chunk.text))
        self._feature_cache[hit.rowid] = value
        if len(self._feature_cache) > 20_000:
            self._feature_cache.popitem(last=False)
        return value

    @staticmethod
    def _hit_from_row(row: sqlite3.Row, lexical_score: float | None = None) -> SQLiteHit:
        return SQLiteHit(
            rowid=int(row["rowid"]),
            chunk=Chunk(
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                title=row["title"],
                heading=row["heading"],
                text=row["text"],
                metadata=json.loads(row["metadata"]),
            ),
            lexical_score=float(lexical_score if lexical_score is not None else row["lexical_score"]),
        )

    def alignment_features_for_hit(self, hit: SQLiteHit) -> tuple[set[str], set[str], set[str]]:
        cached = self._alignment_cache.get(hit.rowid)
        if cached is not None:
            self._alignment_cache.move_to_end(hit.rowid)
            return cached
        value = (set(words(hit.chunk.text, remove_stopwords=True)), citations(hit.chunk.text), numbers(hit.chunk.text))
        self._alignment_cache[hit.rowid] = value
        if len(self._alignment_cache) > 20_000:
            self._alignment_cache.popitem(last=False)
        return value

    def _ordered_query_terms(self, query: str) -> list[str]:
        tokens = list(dict.fromkeys(words(query, remove_stopwords=True)))
        if self._has_vocab and tokens:
            placeholders = ",".join("?" for _ in tokens[:80])
            rows = self.connection.execute(
                f"SELECT term, doc FROM temp.legalqa_vocab WHERE term IN ({placeholders})",
                tokens[:80],
            ).fetchall()
            document_frequency = {row["term"]: int(row["doc"]) for row in rows}
            known = [token for token in tokens if token in document_frequency]
            # Rarest terms make FTS AND queries both faster and more discriminative.
            known.sort(key=lambda token: (document_frequency[token], tokens.index(token)))
            return known + [token for token in tokens if token not in document_frequency]
        return tokens

    def search_documents(
        self,
        query: str,
        top_k: int = 20,
        broad_fallback: bool = True,
    ) -> list[tuple[str, int, float]]:
        terms = self._ordered_query_terms(query)
        expressions = _fts_queries(terms)
        if not expressions or top_k <= 0:
            return []
        if not broad_fallback:
            strict = expressions[:-1]
            if len(terms) >= 2:
                strict.append(" AND ".join(_quote_fts(token) for token in terms[:2]))
            expressions = list(dict.fromkeys(strict)) or expressions
        collected_documents: dict[str, tuple[int, float]] = {}
        for stage, expression in enumerate(expressions):
            rows = self.connection.execute(
                """
                SELECT d.rowid, d.doc_id, bm25(documents_fts, 0.25, 1.0) AS lexical_rank
                FROM documents_fts
                JOIN documents AS d ON d.rowid = documents_fts.rowid
                WHERE documents_fts MATCH ?
                ORDER BY lexical_rank ASC, d.rowid ASC
                LIMIT ?
                """,
                (expression, top_k),
            ).fetchall()
            stage_boost = float(len(expressions) - stage)
            for row in rows:
                doc_id = str(row["doc_id"])
                if doc_id not in collected_documents:
                    collected_documents[doc_id] = (
                        len(collected_documents) + 1,
                        stage_boost + float(-row["lexical_rank"]),
                    )
            if not broad_fallback and rows:
                break
            if len(collected_documents) >= top_k:
                break
            # Only execute the broad OR fallback when strict/grouped retrieval is too sparse.
            if stage == len(expressions) - 2 and len(collected_documents) >= min(8, top_k):
                break
        return [
            (doc_id, rank, score)
            for doc_id, (rank, score) in sorted(collected_documents.items(), key=lambda item: item[1][0])
        ][:top_k]

    def chunks_for_documents(self, doc_ids: list[str]) -> list[SQLiteHit]:
        if not doc_ids:
            return []
        placeholders = ",".join("?" for _ in doc_ids)
        rows = self.connection.execute(
            f"SELECT rowid,chunk_id,doc_id,title,heading,text,metadata FROM chunks WHERE doc_id IN ({placeholders})",
            doc_ids,
        ).fetchall()
        rank = {doc_id: index for index, doc_id in enumerate(doc_ids, 1)}
        return [self._hit_from_row(row, 1.0 / (10 + rank[str(row["doc_id"])])) for row in rows]

    def hits_for_rowids(self, rowids: list[int]) -> list[SQLiteHit]:
        if not rowids:
            return []
        result: dict[int, SQLiteHit] = {}
        # Stay below SQLite's variable limit on older Kaggle images.
        for start in range(0, len(rowids), 800):
            batch = rowids[start : start + 800]
            placeholders = ",".join("?" for _ in batch)
            rows = self.connection.execute(
                f"SELECT rowid,chunk_id,doc_id,title,heading,text,metadata "
                f"FROM chunks WHERE rowid IN ({placeholders})",
                batch,
            ).fetchall()
            for row in rows:
                result[int(row["rowid"])] = self._hit_from_row(row, 0.0)
        return [result[rowid] for rowid in rowids if rowid in result]

    def iter_chunks(self, start_rowid: int = 0, batch_size: int = 256):
        cursor = int(start_rowid)
        while True:
            rows = self.connection.execute(
                "SELECT rowid,chunk_id,doc_id,title,heading,text,metadata "
                "FROM chunks WHERE rowid > ? ORDER BY rowid LIMIT ?",
                (cursor, int(batch_size)),
            ).fetchall()
            if not rows:
                break
            hits = [self._hit_from_row(row, 0.0) for row in rows]
            yield hits
            cursor = hits[-1].rowid

    def chunk_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def search_chunks(self, query: str, top_k: int = 100) -> list[SQLiteHit]:
        """Global chunk-level BM25, independent of document pre-filtering."""
        terms = self._ordered_query_terms(query)
        expressions = _fts_queries(terms, max_terms=18)
        if not expressions or top_k <= 0:
            return []
        collected: dict[int, SQLiteHit] = {}
        for stage, expression in enumerate(expressions):
            rows = self.connection.execute(
                """
                SELECT c.rowid,c.chunk_id,c.doc_id,c.title,c.heading,c.text,c.metadata,
                       bm25(chunks_fts, 0.20, 0.35, 1.0) AS lexical_rank
                FROM chunks_fts
                JOIN chunks AS c ON c.rowid = chunks_fts.rowid
                WHERE chunks_fts MATCH ?
                ORDER BY lexical_rank ASC, c.rowid ASC
                LIMIT ?
                """,
                (expression, int(top_k)),
            ).fetchall()
            stage_boost = float(len(expressions) - stage)
            for row in rows:
                rowid = int(row["rowid"])
                if rowid not in collected:
                    collected[rowid] = self._hit_from_row(
                        row, stage_boost + float(-row["lexical_rank"])
                    )
            if len(collected) >= top_k:
                break
        result = sorted(collected.values(), key=lambda hit: (-hit.lexical_score, hit.rowid))
        return result[:top_k]

    def rank_chunks_in_documents(
        self,
        query: str,
        doc_ids: list[str],
        top_per_document: int = 2,
    ) -> list[SQLiteHit]:
        query_terms = set(legal_tokens(query))
        query_docs = document_numbers(query)
        query_articles = citations(query)
        grouped: dict[str, list[SQLiteHit]] = {}
        for hit in self.chunks_for_documents(doc_ids):
            chunk_terms, chunk_articles, _ = self.features_for_hit(hit)
            chunk_docs = document_numbers(f"{hit.chunk.title}\n{hit.chunk.text}")
            score = (
                1.7 * len(query_terms & chunk_terms) / max(1, len(query_terms))
                + 0.65 * len(query_docs & chunk_docs)
                + 0.30 * len(query_articles & chunk_articles)
            )
            grouped.setdefault(hit.chunk.doc_id, []).append(
                SQLiteHit(hit.rowid, hit.chunk, score)
            )
        result = []
        for doc_id in doc_ids:
            ranked = sorted(grouped.get(doc_id, []), key=lambda hit: (-hit.lexical_score, hit.rowid))
            result.extend(ranked[:top_per_document])
        return result

    def neighbor_chunks(self, chunk_id: str, radius: int = 1) -> list[SQLiteHit]:
        row = self.connection.execute(
            "SELECT rowid,doc_id,metadata FROM chunks WHERE chunk_id=?", (chunk_id,)
        ).fetchone()
        if not row:
            return []
        metadata = json.loads(row["metadata"])
        section = metadata.get("_section_index")
        window = metadata.get("_window_index")
        if section is None or window is None:
            rows = self.connection.execute(
                "SELECT rowid,chunk_id,doc_id,title,heading,text,metadata FROM chunks "
                "WHERE doc_id=? AND rowid BETWEEN ? AND ? ORDER BY rowid",
                (row["doc_id"], max(1, int(row["rowid"]) - radius), int(row["rowid"]) + radius),
            ).fetchall()
        else:
            rows = self.connection.execute(
                "SELECT rowid,chunk_id,doc_id,title,heading,text,metadata FROM chunks WHERE doc_id=?",
                (row["doc_id"],),
            ).fetchall()
            rows = [
                candidate for candidate in rows
                if (candidate_meta := json.loads(candidate["metadata"])).get("_section_index") == section
                and abs(int(candidate_meta.get("_window_index", 0)) - int(window)) <= radius
            ]
        return [self._hit_from_row(candidate, 0.0) for candidate in rows]

    def relation_edges(self, doc_ids: list[str], limit_per_document: int = 8) -> list[RelationEdge]:
        if not doc_ids:
            return []
        result: list[RelationEdge] = []
        for doc_id in doc_ids:
            outgoing = self.connection.execute(
                "SELECT source_doc_id,target_doc_id,relation_type,confidence,evidence "
                "FROM relations WHERE source_doc_id=? "
                "ORDER BY confidence DESC, relation_type, target_doc_id LIMIT ?",
                (doc_id, int(limit_per_document)),
            ).fetchall()
            incoming = self.connection.execute(
                "SELECT source_doc_id,target_doc_id,relation_type,confidence,evidence "
                "FROM relations WHERE target_doc_id=? "
                "ORDER BY confidence DESC, relation_type, source_doc_id LIMIT ?",
                (doc_id, int(limit_per_document)),
            ).fetchall()
            result.extend(
                RelationEdge(
                    row["source_doc_id"], row["target_doc_id"], row["relation_type"],
                    float(row["confidence"]), row["evidence"], "outgoing",
                )
                for row in outgoing
            )
            result.extend(
                RelationEdge(
                    row["source_doc_id"], row["target_doc_id"], row["relation_type"],
                    float(row["confidence"]), row["evidence"], "incoming",
                )
                for row in incoming
            )
        unique: dict[tuple[str, str, str, str], RelationEdge] = {}
        for edge in result:
            key = (edge.source_doc_id, edge.target_doc_id, edge.relation_type, edge.direction)
            if key not in unique or edge.confidence > unique[key].confidence:
                unique[key] = edge
        return sorted(
            unique.values(),
            key=lambda edge: (-edge.confidence, edge.relation_type, edge.neighbor_doc_id),
        )

    def relation_count(self, doc_ids: list[str]) -> int:
        if not doc_ids:
            return 0
        placeholders = ",".join("?" for _ in doc_ids)
        row = self.connection.execute(
            f"SELECT COUNT(*) FROM relations WHERE source_doc_id IN ({placeholders}) "
            f"OR target_doc_id IN ({placeholders})",
            [*doc_ids, *doc_ids],
        ).fetchone()
        return int(row[0])

    def obsolete_documents(self, doc_ids: list[str]) -> set[str]:
        """Return documents superseded/repealed by another official document."""
        if not doc_ids:
            return set()
        result: set[str] = set()
        for start in range(0, len(doc_ids), 800):
            batch = doc_ids[start : start + 800]
            placeholders = ",".join("?" for _ in batch)
            rows = self.connection.execute(
                f"SELECT DISTINCT target_doc_id FROM relations "
                f"WHERE relation_type IN ('REPLACES','REPEALS') "
                f"AND target_doc_id IN ({placeholders})",
                batch,
            ).fetchall()
            result.update(str(row[0]) for row in rows)
        return result

    def search(self, query: str, top_k: int = 80) -> list[SQLiteHit]:
        document_limit = min(30, max(12, top_k // 3))
        document_hits = self.search_documents(query, document_limit, broad_fallback=True)
        collected_documents = {doc_id: (rank, score) for doc_id, rank, score in document_hits}
        if not collected_documents:
            return []
        chunk_hits = self.chunks_for_documents(list(collected_documents))
        query_terms = set(legal_tokens(query))
        query_citations, query_numbers = citations(query), numbers(query)
        hits = []
        for chunk_hit in chunk_hits:
            text = chunk_hit.chunk.text
            chunk_terms, chunk_citations, chunk_numbers = self.features_for_hit(chunk_hit)
            doc_rank, document_score = collected_documents[chunk_hit.chunk.doc_id]
            overlap = len(query_terms & chunk_terms) / max(1, len(query_terms))
            citation_score = len(query_citations & chunk_citations)
            number_score = len(query_numbers & chunk_numbers)
            lexical_score = (
                document_score + 1.6 * overlap + 0.18 * citation_score
                + 0.04 * number_score + 0.25 / (10 + doc_rank)
            )
            hits.append(
                SQLiteHit(
                    rowid=chunk_hit.rowid,
                    chunk=chunk_hit.chunk,
                    lexical_score=lexical_score,
                )
            )
        hits.sort(key=lambda hit: (-hit.lexical_score, hit.rowid))
        return hits[:top_k]


def build_sqlite_store(
    config: dict[str, Any],
    corpus_path: str | Path,
    database_path: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    database_path = Path(database_path)
    fingerprint = corpus_fingerprint(corpus_path)
    if database_path.exists() and not force:
        try:
            with SQLiteLegalStore(database_path, readonly=True) as store:
                manifest = store.manifest()
        except (OSError, sqlite3.DatabaseError) as exc:
            manifest = {}
            print(f"[SQLite] index cũ bị hỏng/dở dang, sẽ dựng lại: {exc}", flush=True)
        if (
            manifest.get("status") == "complete"
            and manifest.get("schema_version") == STORE_SCHEMA_VERSION
            and manifest.get("corpus_fingerprint") == fingerprint
        ):
            return {**manifest, "reused": True, "database": str(database_path.resolve())}
        if (
            manifest.get("status") == "complete"
            and manifest.get("corpus_fingerprint") == fingerprint
            and manifest.get("schema_version") != STORE_SCHEMA_VERSION
        ):
            print(
                f"[SQLite] nâng schema {manifest.get('schema_version')} -> "
                f"{STORE_SCHEMA_VERSION}; tự động dựng lại index từ corpus chính thức.",
                flush=True,
            )
        elif manifest.get("status") == "complete":
            raise RuntimeError(
                f"SQLite index hoàn chỉnh nhưng không khớp corpus: {database_path}. "
                "Dùng --force-index để xác nhận tạo lại."
            )
        print("[SQLite] phát hiện index chưa hoàn tất; tự động dựng lại an toàn.", flush=True)

    database_path.parent.mkdir(parents=True, exist_ok=True)
    building_path = database_path.with_name(f".{database_path.name}.building")
    if building_path.exists():
        building_path.unlink()

    with SQLiteLegalStore(building_path) as store:
        db = store.connection
        db.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=MEMORY;
            CREATE TABLE manifest (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE documents (
                rowid INTEGER PRIMARY KEY,
                doc_id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                metadata TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE documents_fts USING fts5(
                title, passage, content='', tokenize='unicode61 remove_diacritics 0'
            );
            CREATE TABLE chunks (
                rowid INTEGER PRIMARY KEY,
                chunk_id TEXT UNIQUE NOT NULL,
                doc_id TEXT NOT NULL,
                title TEXT NOT NULL,
                heading TEXT NOT NULL,
                text TEXT NOT NULL,
                metadata TEXT NOT NULL
            );
            CREATE INDEX chunks_doc_id_idx ON chunks(doc_id);
            CREATE TABLE document_aliases (
                alias TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                UNIQUE(alias, doc_id)
            );
            CREATE INDEX document_aliases_alias_idx ON document_aliases(alias);
            CREATE TABLE relations (
                source_doc_id TEXT NOT NULL,
                target_doc_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                evidence TEXT NOT NULL,
                UNIQUE(source_doc_id, target_doc_id, relation_type)
            );
            CREATE INDEX relations_source_idx ON relations(source_doc_id);
            CREATE INDEX relations_target_idx ON relations(target_doc_id);
            """
        )
        inserted = 0
        documents = 0
        pending = []
        aliases_by_doc: dict[str, set[str]] = {}
        pending_mentions = []
        for document in iter_corpus(corpus_path, config):
            documents += 1
            own_aliases = document_aliases(document.title, document.text)
            aliases_by_doc[document.doc_id] = own_aliases
            pending_mentions.extend(
                (document.doc_id, mention)
                for mention in extract_relation_mentions(document.text, own_aliases)
            )
            document_metadata = json.dumps(document.metadata, ensure_ascii=False, separators=(",", ":"))
            cursor = db.execute(
                "INSERT INTO documents(doc_id,title,metadata) VALUES(?,?,?)",
                (document.doc_id, document.title, document_metadata),
            )
            db.execute(
                "INSERT INTO documents_fts(rowid,title,passage) VALUES(?,?,?)",
                (cursor.lastrowid, document.title, document.text),
            )
            for chunk in chunk_documents([document], config):
                pending.append(
                    (
                        chunk.chunk_id, chunk.doc_id, chunk.title, chunk.heading, chunk.text,
                        json.dumps(chunk.metadata, ensure_ascii=False, separators=(",", ":")),
                    )
                )
                if len(pending) >= 500:
                    db.executemany(
                        "INSERT INTO chunks(chunk_id,doc_id,title,heading,text,metadata) VALUES(?,?,?,?,?,?)",
                        pending,
                    )
                    inserted += len(pending)
                    pending.clear()
            if documents % 500 == 0:
                db.commit()
                print(f"[SQLite] documents={documents:,} chunks={inserted + len(pending):,}", flush=True)
        if pending:
            db.executemany(
                "INSERT INTO chunks(chunk_id,doc_id,title,heading,text,metadata) VALUES(?,?,?,?,?,?)", pending
            )
            inserted += len(pending)
        print("[SQLite] building global chunk FTS and legal relation graph", flush=True)
        db.execute(
            "CREATE VIRTUAL TABLE chunks_fts USING fts5("
            "title, heading, text, content='chunks', content_rowid='rowid', "
            "tokenize='unicode61 remove_diacritics 0')"
        )
        db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")

        alias_rows = [
            (alias, doc_id, "document_number")
            for doc_id, aliases in aliases_by_doc.items()
            for alias in sorted(aliases)
        ]
        if alias_rows:
            db.executemany(
                "INSERT OR IGNORE INTO document_aliases(alias,doc_id,kind) VALUES(?,?,?)",
                alias_rows,
            )
        alias_to_docs: dict[str, list[str]] = {}
        for alias, doc_id, _ in alias_rows:
            alias_to_docs.setdefault(alias, []).append(doc_id)
        relation_rows: dict[tuple[str, str, str], tuple[float, str]] = {}
        for source_doc_id, mention in pending_mentions:
            for matched_doc_id in alias_to_docs.get(mention.target_alias, []):
                if matched_doc_id == source_doc_id:
                    continue
                source, target = (
                    (matched_doc_id, source_doc_id)
                    if mention.reverse
                    else (source_doc_id, matched_doc_id)
                )
                key = (source, target, mention.relation_type)
                previous = relation_rows.get(key)
                if previous is None or mention.confidence > previous[0]:
                    relation_rows[key] = (mention.confidence, mention.evidence[:500])
        if relation_rows:
            db.executemany(
                "INSERT OR REPLACE INTO relations("
                "source_doc_id,target_doc_id,relation_type,confidence,evidence"
                ") VALUES(?,?,?,?,?)",
                [
                    (source, target, relation_type, confidence, evidence)
                    for (source, target, relation_type), (confidence, evidence)
                    in relation_rows.items()
                ],
            )
        values = {
            "status": "complete",
            "schema_version": STORE_SCHEMA_VERSION,
            "corpus_fingerprint": fingerprint,
            "documents": str(documents),
            "chunks": str(inserted),
            "aliases": str(len(alias_rows)),
            "relations": str(len(relation_rows)),
        }
        db.executemany("INSERT INTO manifest(key,value) VALUES(?,?)", values.items())
        db.commit()
    try:
        os.replace(building_path, database_path)
    except PermissionError:
        # Windows does not reliably allow os.replace() over an existing SQLite
        # file. The complete replacement already exists before this fallback.
        if os.name != "nt" or not database_path.exists():
            raise
        database_path.unlink()
        os.replace(building_path, database_path)
    return {
        "status": "complete", "schema_version": STORE_SCHEMA_VERSION,
        "corpus_fingerprint": fingerprint, "documents": documents,
        "chunks": inserted, "database": str(database_path.resolve()), "reused": False,
        "aliases": len(alias_rows), "relations": len(relation_rows),
    }


def _features(text: str) -> tuple[set[str], set[str], set[str]]:
    return set(words(text, remove_stopwords=True)), citations(text), numbers(text)


def _alignment_score(
    answer: tuple[set[str], set[str], set[str]],
    passage: tuple[set[str], set[str], set[str]],
) -> float:
    answer_tokens, answer_citations, answer_numbers = answer
    passage_tokens, passage_citations, passage_numbers = passage
    lexical = overlap_ratio(answer_tokens, passage_tokens)
    citation = len(answer_citations & passage_citations) / len(answer_citations) if answer_citations else 0.0
    numeric = len(answer_numbers & passage_numbers) / len(answer_numbers) if answer_numbers else 0.0
    return 0.64 * lexical + 0.26 * citation + 0.10 * numeric


def prepare_sqlite_weak_pairs(
    train_path: str | Path,
    database_path: str | Path,
    output_path: str | Path,
    top_k: int = 40,
    negatives: int = 3,
    limit: int | None = None,
) -> dict[str, Any]:
    records = load_qa(train_path, require_answers=True)
    if limit is not None:
        records = records[:limit]
    rows = []
    skipped = 0
    with SQLiteLegalStore(database_path, readonly=True) as store:
        for record_number, record in enumerate(records, 1):
            answer = record.answer or ""
            # The question adds topic terms while citation-bearing answer terms anchor the legal document.
            citation_terms = " ".join(sorted(citations(answer)))
            numeric_terms = " ".join(sorted(numbers(answer))[:10])
            query = f"{record.question}\n{citation_terms}\n{numeric_terms}"
            document_k = min(4, max(3, top_k // 12))
            document_hits = store.search_documents(query, top_k=document_k, broad_fallback=False)
            if not document_hits:
                document_hits = store.search_documents(record.question, top_k=document_k, broad_fallback=True)
            hits = store.chunks_for_documents([doc_id for doc_id, _, _ in document_hits])
            answer_features = _features(answer)
            rescored = [
                (hit, _alignment_score(answer_features, store.alignment_features_for_hit(hit))) for hit in hits
            ]
            rescored.sort(key=lambda item: (-item[1], -item[0].lexical_score, item[0].rowid))
            if not rescored or rescored[0][1] <= 0:
                skipped += 1
                continue
            positive, positive_score = rescored[0]
            supporting = []
            supporting_docs = set()
            for hit, score in rescored:
                if score < max(0.12, positive_score * 0.45):
                    continue
                if hit.chunk.doc_id in supporting_docs:
                    continue
                supporting.append(hit)
                supporting_docs.add(hit.chunk.doc_id)
                if len(supporting) >= 3:
                    break
            hard_negatives = [
                hit.chunk.text for hit, score in rescored[1:]
                if hit.chunk.doc_id != positive.chunk.doc_id and score < max(0.20, positive_score * 0.68)
            ][:negatives]
            if not hard_negatives:
                hard_negatives = [hit.chunk.text for hit, _ in rescored[-negatives:] if hit.rowid != positive.rowid]
            rows.append(
                {
                    "question_id": record.question_id,
                    "question": record.question,
                    "answer": answer,
                    "positive": positive.chunk.text,
                    "positive_chunk_id": positive.chunk.chunk_id,
                    "positive_doc_id": positive.chunk.doc_id,
                    "positive_evidence": [hit.chunk.text for hit in supporting],
                    "positive_chunk_ids": [hit.chunk.chunk_id for hit in supporting],
                    "alignment_score": positive_score,
                    "negatives": hard_negatives,
                }
            )
            if record_number % 250 == 0:
                print(
                    f"[Weak labels] processed={record_number:,}/{len(records):,} pairs={len(rows):,} skipped={skipped:,}",
                    flush=True,
                )
    write_jsonl(output_path, rows)
    scores = sorted(row["alignment_score"] for row in rows)
    return {
        "records": len(records), "pairs": len(rows), "skipped": skipped,
        "alignment_median": scores[len(scores) // 2] if scores else 0.0,
        "alignment_p10": scores[max(0, round(len(scores) * 0.10) - 1)] if scores else 0.0,
        "output": str(Path(output_path).resolve()),
    }
