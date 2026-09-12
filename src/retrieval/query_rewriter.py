import logging

from openai import OpenAI

from src.settings import Settings

logger = logging.getLogger(__name__)

REWRITER_SYSTEM_PROMPT = (
    "You are a search query optimizer for financial stability report retrieval. "
    "Rewrite the user's question as a concise, keyword-rich search query. "
    "Output only the rewritten query. No explanation. No punctuation at the end."
)


class QueryRewriter:
    def __init__(self, cfg: Settings) -> None:
        self._cfg = cfg
        self._client = OpenAI(
            api_key=cfg.saia.api_key,
            base_url=cfg.saia.base_url,
        )
        self._cache: dict[str, str] = {}

    def rewrite(self, query: str) -> str:
        """Rewrite query for retrieval. Caches results within a session."""
        if query in self._cache:
            logger.debug("Cache hit for query: %s", query)
            return self._cache[query]

        response = self._client.chat.completions.create(
            model=self._cfg.saia.model,
            temperature=self._cfg.saia.temperature,
            max_tokens=self._cfg.saia.rewriter_max_tokens,
            messages=[
                {"role": "system", "content": REWRITER_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
        )
        # content is None on a filtered/empty response — fall back to the
        # original query so retrieval still runs
        content = response.choices[0].message.content or query
        rewritten = content.strip() or query
        logger.debug("Rewrite: %r → %r", query, rewritten)
        self._cache[query] = rewritten
        return rewritten
