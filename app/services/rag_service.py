import csv
import logging
import re
import unicodedata
from typing import List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentChunk:
    def __init__(self, content: str, source_file: str, metadata: dict = None):
        self.content = content.strip()
        self.source_file = source_file
        self.metadata = metadata or {}
        self.keywords = self._extract_keywords(content)

    def _extract_keywords(self, text: str) -> set:
        words = re.findall(r'\b\w{3,}\b', text.lower())
        stopwords = {
            'les', 'des', 'une', 'que', 'qui', 'pour', 'par', 'sur', 'dans',
            'avec', 'est', 'sont', 'the', 'and', 'for', 'that', 'this', 'with',
            'has', 'have', 'from', 'aux', 'ces', 'leur', 'leurs', 'tout',
            'mais', 'plus', 'aussi', 'tres', 'bien', 'etre', 'avoir', 'fait',
            'comme', 'meme', 'alors', 'donc', 'car', 'pas', 'ses', 'son', 'elle'
        }
        return set(words) - stopwords


class RAGService:
    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.is_initialized = False
        self.stats = {}

    # Mapping pays → mots-clés présents dans les noms de dossiers/fichiers
    COUNTRY_MAPPING = {
        "senegal":    ["acl_sn", "faq_data", "citizenlab", "mission_citizen", "programme", "scrap_site", "citizenlab_team", "citizenlab_rag"],
        "sénégal":    ["acl_sn", "faq_data", "citizenlab", "mission_citizen", "programme", "scrap_site", "citizenlab_team", "citizenlab_rag"],
        "benin":      ["acl_benin"],
        "bénin":      ["acl_benin"],
        "cameroun":   ["acl_cameroun"],
        "cameroon":   ["acl_cameroun"],
        "chad":       ["acl_chad"],
        "tchad":      ["acl_chad"],
        "guinee":     ["acl_guinee"],
        "guinée":     ["acl_guinee"],
        "madagascar": ["acl_madagascar"],
        "mauritania": ["acl_mauritania"],
        "mauritanie": ["acl_mauritania"],
    }

    def _normalize(self, text: str) -> str:
        """Supprime les accents et met en minuscules"""
        text = text.lower().strip()
        nfkd = unicodedata.normalize('NFKD', text)
        return ''.join(c for c in nfkd if not unicodedata.combining(c))

    def _get_country_keywords(self, country: str) -> List[str]:
        """Retourne les mots-clés fichiers/dossiers associés à un pays"""
        country_norm = self._normalize(country)
        # Cherche dans le mapping avec normalisation
        for key, keywords in self.COUNTRY_MAPPING.items():
            if self._normalize(key) == country_norm:
                return keywords
        # Fallback : utilise le nom tel quel
        return [country_norm]

    def _detect_country_in_query(self, query_normalized: str) -> List[str]:
        """Détecte automatiquement un pays mentionné dans la question"""
        for key, keywords in self.COUNTRY_MAPPING.items():
            if self._normalize(key) in query_normalized:
                return keywords
        return []

    def initialize(self, knowledge_base_dir: str = "knowledge_base"):
        kb_path = Path(knowledge_base_dir)

        if not kb_path.exists():
            logger.warning(f"⚠️  Dossier knowledge_base non trouvé : {kb_path.absolute()}")
            self.is_initialized = True
            return

        total_loaded = 0
        file_counts = {}

        for file_path in kb_path.rglob("*"):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                chunks_loaded = 0
                try:
                    if ext == ".csv":
                        chunks_loaded = self._load_csv(file_path)
                    elif ext == ".pdf":
                        chunks_loaded = self._load_pdf(file_path)
                    elif ext == ".txt":
                        chunks_loaded = self._load_txt(file_path)

                    if chunks_loaded > 0:
                        file_counts[file_path.name] = chunks_loaded
                        total_loaded += chunks_loaded
                        logger.info(f"  ✅ {file_path.name} → {chunks_loaded} chunks")

                except Exception as e:
                    logger.error(f"  ❌ Erreur {file_path.name}: {e}")

        self.stats = {
            "total_chunks": total_loaded,
            "files_loaded": len(file_counts),
            "file_details": file_counts
        }
        self.is_initialized = True
        logger.info(f"📊 Total : {total_loaded} chunks depuis {len(file_counts)} fichiers")

    def _load_csv(self, file_path: Path) -> int:
        count = 0
        folder = file_path.parent.name
        encodings = ['utf-8', 'latin-1', 'utf-8-sig']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        text_parts = [f"{k}: {v}" for k, v in row.items() if v and str(v).strip()]
                        if text_parts:
                            content = " | ".join(text_parts)
                            chunks = self._chunk_text(content, file_path.name, folder)
                            self.chunks.extend(chunks)
                            count += len(chunks)
                break
            except Exception:
                continue
        return count

    def _load_pdf(self, file_path: Path) -> int:
        count = 0
        folder = file_path.parent.name
        try:
            import pypdf
            with open(file_path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        chunks = self._chunk_text(
                            text, file_path.name, folder,
                            extra_meta={"page": page_num + 1}
                        )
                        self.chunks.extend(chunks)
                        count += len(chunks)
        except ImportError:
            logger.warning("pypdf non installé, PDFs ignorés.")
        except Exception as e:
            logger.error(f"Erreur PDF {file_path.name}: {e}")
        return count

    def _load_txt(self, file_path: Path) -> int:
        folder = file_path.parent.name
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            chunks = self._chunk_text(text, file_path.name, folder)
            self.chunks.extend(chunks)
            return len(chunks)
        except Exception as e:
            logger.error(f"Erreur TXT {file_path.name}: {e}")
            return 0

    def _chunk_text(self, text: str, source: str, folder: str, chunk_size: int = 400, overlap: int = 50, extra_meta: dict = None) -> List[DocumentChunk]:
        text = text.strip()
        meta = {"folder": folder}
        if extra_meta:
            meta.update(extra_meta)

        if len(text) <= chunk_size:
            return [DocumentChunk(text, source, meta)]

        chunks = []
        words = text.split()
        current_chunk = []
        current_len = 0

        for word in words:
            current_chunk.append(word)
            current_len += len(word) + 1
            if current_len >= chunk_size:
                chunks.append(DocumentChunk(" ".join(current_chunk), source, meta))
                current_chunk = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_len = sum(len(w) + 1 for w in current_chunk)

        if current_chunk:
            chunks.append(DocumentChunk(" ".join(current_chunk), source, meta))

        return chunks

    def search(self, query: str, top_k: int = 5, country_filter: Optional[str] = None) -> List[Tuple[DocumentChunk, float]]:
        if not self.chunks:
            return []

        stopwords = {
            'les', 'des', 'une', 'que', 'qui', 'pour', 'the', 'and', 'for',
            'quels', 'comment', 'what', 'how', 'est', 'sont', 'avec', 'dans',
            'cest', 'quoi', 'vous', 'nous', 'ils', 'elles', 'votre', 'notre'
        }

        query_normalized = self._normalize(query)
        query_keywords = set(re.findall(r'\b\w{3,}\b', query_normalized)) - stopwords

        # Détermine les mots-clés pays à utiliser
        if country_filter:
            # Pays fourni explicitement → utilise le mapping
            country_keywords = self._get_country_keywords(country_filter)
            logger.debug(f"🌍 Filtre pays '{country_filter}' → keywords: {country_keywords}")
        else:
            # Détection automatique du pays dans la question
            country_keywords = self._detect_country_in_query(query_normalized)
            if country_keywords:
                logger.debug(f"🔍 Pays détecté dans la query → keywords: {country_keywords}")

        scored_chunks = []

        for chunk in self.chunks:
            folder_normalized = self._normalize(chunk.metadata.get("folder", ""))
            source_normalized = self._normalize(chunk.source_file)

            # Filtre par pays si des mots-clés ont été détectés
            if country_keywords:
                match = any(
                    kw in folder_normalized or kw in source_normalized
                    for kw in country_keywords
                )
                if not match:
                    continue

            # Calcul du score
            if not query_keywords:
                score = 0.1
            else:
                chunk_content_normalized = self._normalize(chunk.content)
                chunk_keywords_normalized = set(re.findall(r'\b\w{3,}\b', chunk_content_normalized))
                common = query_keywords & chunk_keywords_normalized
                score = len(common) / len(query_keywords)
                for keyword in query_keywords:
                    if keyword in chunk_content_normalized:
                        score += 0.05

            if score > 0:
                scored_chunks.append((chunk, min(score, 1.0)))

        # Fallback : si aucun résultat avec filtre pays → cherche sans filtre
        if not scored_chunks and country_keywords:
            logger.warning(f"⚠️ Aucun résultat avec filtre pays, recherche globale...")
            return self.search(query=query, top_k=top_k, country_filter=None)

        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]

    def format_context(self, results: List[Tuple[DocumentChunk, float]]) -> str:
        if not results:
            return "Aucun contexte trouvé dans la knowledge base."
        parts = [
            f"[Source {i} - {chunk.source_file}]\n{chunk.content}"
            for i, (chunk, _) in enumerate(results, 1)
        ]
        return "\n\n---\n\n".join(parts)

    def get_stats(self) -> dict:
        return self.stats