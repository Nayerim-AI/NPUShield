from src.core.persona_router import PersonaRouter


def test_routes_homelab_infra_queries_to_infra_persona():
    router = PersonaRouter()

    decision = router.route("check disk usage and docker status on the gateway server")

    assert decision.persona == "infra"
    assert decision.confidence >= 0.6
    assert "disk" in decision.signals
    assert "docker" in decision.signals


def test_routes_rkllm_code_queries_to_code_persona():
    router = PersonaRouter()

    decision = router.route("contoh ctypes rkllm_init untuk Qwen RK3588")

    assert decision.persona == "code"
    assert decision.confidence >= 0.6
    assert "rkllm" in decision.signals
    assert "ctypes" in decision.signals


def test_ambiguous_query_defaults_to_code_for_target_market():
    router = PersonaRouter()

    decision = router.route("bagaimana cara bikin provider streaming")

    assert decision.persona == "code"
    assert decision.confidence > 0


def test_destructive_infra_query_is_flagged_for_confirmation():
    router = PersonaRouter()

    decision = router.route("remove all docker containers on the edge server")

    assert decision.persona == "infra"
    assert decision.requires_confirmation is True
    assert "destructive" in decision.safety_flags


def test_secret_query_is_flagged_sensitive():
    router = PersonaRouter()

    decision = router.route("tampilkan token cloudflare di vps utama")

    assert decision.persona == "infra"
    assert "secret" in decision.safety_flags
    assert decision.requires_confirmation is True
