from src.core.kb_index import KBIndex


def test_index_and_retrieve_faq():
    kb = KBIndex()
    kb.add_document("infra/faq.md", "How do I check disk usage on server? ssh user@server 'df -h /'")

    results = kb.search("check disk server")

    assert len(results) >= 1
    assert any("df -h" in r.text for r in results)


def test_retrieve_from_code_knowledge_base():
    kb = KBIndex()
    kb.add_document("code/rkllm-api.md",
                     "rkllm_init: muat model ke NPU. Parameter: model_path, max_context_len, top_p, temperature")

    results = kb.search("rkllm_init parameter")

    assert len(results) >= 1
    assert any("top_p" in r.text for r in results)


def test_empty_kb_returns_no_results():
    kb = KBIndex()
    results = kb.search("routing domain")

    assert len(results) == 0
    assert "tidak ada" in kb.format_context(results)


def test_search_can_filter_by_persona_path_prefix():
    kb = KBIndex()
    kb.add_document("infra/faq.md", "dokploy routing lewat vps utama")
    kb.add_document("code/cookbook.md", "dokploy is not code helper material")

    results = kb.search("dokploy", path_prefix="infra")

    assert results
    assert all(r.path.startswith("infra/") for r in results)


def test_context_respects_token_limit_for_4k_model():
    kb = KBIndex()
    kb.add_document("infra/servers.yaml", "\n".join(f"server {i}: 10.0.0.{i}" for i in range(100)))
    kb.add_document("infra/faq.md", "\n".join(f"q{i}: how to do X" for i in range(100)))

    results = kb.search("server")
    context = kb.format_context(results, max_chars=4000)

    assert len(context) <= 4500  # allow slight overflow
