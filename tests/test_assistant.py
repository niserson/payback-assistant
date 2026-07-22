"""End-to-end and unit tests for intent detection, retrieval and the agent policy."""

import pytest
from fastapi.testclient import TestClient

from app.catalog import build_catalog
from app.intent import detect, detect_language
from app.main import app
from app.retrieval import BM25Index


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    """Tests must be deterministic: force the rule path even if a key is set."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("VERTEX_PROJECT", raising=False)
    from app import context
    context.reset()


@pytest.fixture(scope="module")
def index():
    return BM25Index(build_catalog())


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ---------- language detection ----------

@pytest.mark.parametrize("query,lang", [
    ("Bitte zeige mir Angebote für günstige Windeln", "de"),
    ("I need stuff for a pasta dinner", "en"),
    ("Where can I find cheap headphones?", "en"),
    ("Suche Shampoo bei dm", "de"),
])
def test_language(query, lang):
    assert detect_language(query) == lang


# ---------- intent detection ----------

@pytest.mark.parametrize("query,intent", [
    ("Bitte zeige mir Angebote für günstige Windeln", "search"),
    ("I need stuff for a pasta dinner", "discovery"),
    ("Was ist besser: Bio-Olivenöl oder normales Olivenöl?", "comparison"),
    ("Ich habe ein Problem mit meinen PAYBACK Punkten", "customer_support"),
    ("My refund has not arrived", "customer_support"),
])
def test_intent(query, intent, index):
    assert detect(query, index.vocabulary()).intent == intent


def test_navigational_partner(index):
    result = detect("Suche Shampoo bei dm", index.vocabulary())
    assert result.partner == "dm"


def test_vague_is_not_specific(index):
    result = detect("Ich brauche mal was Neues", index.vocabulary())
    assert not result.is_specific


def test_unknown_tokens_flagged_for_llm(index):
    # 'fruhstuck' matches the index, 'spiegelei' does not -> partial understanding.
    result = detect("spiegelei fur fruhstuck", index.vocabulary())
    assert result.is_specific
    assert "spiegelei" in result.unknown_tokens
    assert "fur" not in result.unknown_tokens  # umlaut-less filler is a stopword


# ---------- retrieval (incl. cross-lingual + cold start) ----------

def test_german_query_finds_diapers(index):
    hits = index.search("günstige Windeln")
    assert hits and hits[0]["partner"] == "dm" and "Windeln" in hits[0]["name"]


def test_english_query_crosses_language(index):
    hits = index.search("cheap diapers")
    assert hits and "Windeln" in hits[0]["name"]


def test_no_weak_tail_false_positives(index):
    # "günstige" must not drag in unrelated items via the "Gut&Günstig" brand.
    hits = index.search("Bitte zeige mir Angebote für günstige Windeln")
    assert hits and all("Windeln" in h["name"] for h in hits)


def test_shopping_list_fully_understood_by_classifier(index):
    # Umlaut-less typing folds to catalog vocabulary; the unknown compound
    # (Avocadobrot) is flagged for LLM escalation instead of being hand-mapped.
    result = detect("Sussigkeiten, Avocadobrot, Pizza und Kuchen", index.vocabulary())
    assert result.is_specific
    assert result.unknown_tokens == ["avocadobrot"]


def test_shopping_list_returns_products(client):
    body = ask(client, "Sussigkeiten, Avocadobrot, Pizza und Kuchen")
    assert body["engine"] == "classifier" and body["action"]["type"] == "recommend"
    names = " ".join(p["name"] for p in body["products"]).lower()
    assert "pizza" in names
    assert ("schokolade" in names) or ("gummibärchen" in names)
    assert "kuchen" in names


def test_interest_rank_boost(index):
    # Same ambiguous query, different profiles -> the favored category rises to #1
    # (Bio-Honig and Bio-Eier both carry 'bio' in their NAMES, so base scores tie).
    breakfast = index.search("bio", interests={"Frühstück": 100.0})
    dairy = index.search("bio", interests={"Molkerei & Eier": 100.0})
    assert breakfast[0]["category"] == "Frühstück"
    assert dairy[0]["category"] == "Molkerei & Eier"


def test_partner_scoping(index):
    hits = index.search("Shampoo", partner="dm")
    assert hits and all(h["partner"] == "dm" for h in hits)


def test_search_spans_partners(index):
    hits = index.search("pasta tomaten kochbuch", top_k=10)
    assert {h["partner"] for h in hits} >= {"edeka", "amazon"}


# ---------- API / agent policy ----------

def ask(client, query):
    response = client.post("/assist", json={"query": query})
    assert response.status_code == 200
    return response.json()


def test_search_returns_products(client):
    body = ask(client, "Bitte zeige mir Angebote für günstige Windeln")
    assert body["action"]["type"] == "recommend" and body["products"]


def test_discovery_theme_basket(client):
    body = ask(client, "I need stuff for a pasta dinner")
    assert body["intent"] == "discovery" and body["language"] == "en"
    names = " ".join(p["name"] for p in body["products"])
    assert "Spaghetti" in names or "Tomaten" in names


def test_navigational_routes(client):
    body = ask(client, "Suche Shampoo bei dm")
    assert body["action"]["type"] == "route_to_partner"
    assert body["partner_filter"] == "dm"
    assert all(p["partner"] == "dm" for p in body["products"])


def test_vague_asks_clarifying_question(client):
    body = ask(client, "Ich brauche ein Geschenk für jemanden")
    assert body["action"]["type"] == "clarify" and body["clarifying_question"]
    body2 = ask(client, "Ich suche irgendwas Schönes")
    assert body2["action"]["type"] == "clarify" and body2["clarifying_question"]


def test_semantic_net_grounds_unknown_dish(client):
    # LLM off + no lexical match -> EmbeddingGemma cosine net finds the eggs.
    body = client.post("/assist", json={"query": "omelette", "llm_mode": "off"}).json()
    assert body["engine"] == "classifier+semantic"
    assert any("Eier" in p["name"] for p in body["products"])


def test_semantic_net_rejects_out_of_domain(client):
    # Below the similarity threshold -> clarify, no junk products.
    body = client.post("/assist", json={"query": "wie wird das wetter morgen",
                                        "llm_mode": "off"}).json()
    assert body["action"]["type"] == "clarify"


def test_support_handoff(client):
    body = ask(client, "Ich habe ein Problem mit meinen PAYBACK Punkten")
    assert body["intent"] == "customer_support"
    assert body["action"]["type"] == "support_handoff" and not body["products"]


# ---------- evaluation harness ----------

def test_eval_dataset_size_and_coverage():
    from evaluation.dataset import build_dataset
    data = build_dataset()
    assert len(data) >= 300
    intents = {e["intent"] for e in data}
    assert intents == {"search", "discovery", "comparison", "customer_support"}
    assert {e["language"] for e in data} == {"de", "en"}
    assert build_dataset() == data  # deterministic


def test_eval_metrics_thresholds(index):
    from evaluation.dataset import build_dataset
    from evaluation.harness import run
    report = run(build_dataset(), index)
    assert report["intent_accuracy"] >= 0.95
    assert report["language_accuracy"] >= 0.95
    assert report["action_accuracy"] >= 0.95
    assert report["hit@5"] >= 0.95
    assert report["ndcg@5"] >= 0.90


def test_ranking_metrics_math():
    from evaluation.harness import score_ranking
    perfect = score_ranking(["a", "b"], {"a", "b"})
    assert perfect["hit"] == 1.0 and perfect["mrr"] == 1.0 and abs(perfect["ndcg"] - 1.0) < 1e-9
    second = score_ranking(["x", "a"], {"a"})
    assert second["mrr"] == 0.5
    miss = score_ranking(["x", "y"], {"a"})
    assert miss["hit"] == 0.0 and miss["ndcg"] == 0.0


def test_input_validation(client):
    assert client.post("/assist", json={"query": ""}).status_code == 422
    assert client.post("/assist", json={"query": "x" * 501}).status_code == 422


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["products"] > 0


def test_ui_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "PAYBACK Lightweight Assistant" in response.text


def test_optimization_report_served(client):
    response = client.get("/optimization-report")
    assert response.status_code == 200
    assert "thinkingBudget" in response.text


def test_taxonomy_tree_svg(client):
    response = client.get("/taxonomy-tree")
    assert response.status_code == 200
    assert "<svg" in response.text
    assert "Baby &amp; Kind" in response.text
    assert "PMID_" not in response.text  # all connector placeholders resolved


def test_llm_disabled_returns_none():
    from app import llm
    assert not llm.available()
    assert llm.classify("anything") is None


def test_llm_validate_maps_compact_keys():
    from app.llm import _validate
    result = _validate({"i": "search", "l": "de", "p": "dm", "t": "windeln", "c": None})
    assert result == {"intent": "search", "language": "de", "partner": "dm",
                      "search_terms": "windeln", "clarifying_question": None}
    # long-form keys still accepted
    assert _validate({"intent": "search", "language": "en", "partner": None,
                      "search_terms": "eggs", "clarifying_question": None})["search_terms"] == "eggs"
    assert _validate({"i": "nonsense"}) is None


def test_rules_engine_marker(client):
    body = ask(client, "günstige Windeln")
    assert body["engine"] == "classifier"


def test_llm_mode_off_never_calls_llm(client, monkeypatch):
    from app import llm
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "classify", lambda *a, **k: pytest.fail("LLM called despite llm_mode=off"))
    body = client.post("/assist", json={"query": "irgendwas total unbekanntes zeug",
                                        "llm_mode": "off"}).json()
    assert body["engine"] == "classifier"


def test_llm_mode_always_calls_llm_on_known_query(client, monkeypatch):
    from app import llm
    calls = []

    def fake_classify(query, user_context=None, model=None):
        calls.append(model)
        return {"intent": "search", "language": "de", "partner": None,
                "search_terms": "windeln", "clarifying_question": None}

    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "backend", lambda: "test")
    monkeypatch.setattr(llm, "classify", fake_classify)
    body = client.post("/assist", json={"query": "günstige Windeln", "llm_mode": "always",
                                        "model": "gemini-2.5-flash"}).json()
    assert calls == ["gemini-2.5-flash"]
    assert body["engine"].startswith("classifier+gemini-2.5-flash@")


def test_model_allowlist_validated(client):
    response = client.post("/assist", json={"query": "Windeln", "model": "gpt-4o"})
    assert response.status_code == 422


def test_clarify_with_profile_adds_suggestions(client):
    uid = "suggest-user"
    client.post("/assist", json={"query": "Yogamatte", "user_id": uid})
    body = client.post("/assist", json={"query": "irgendwas Schönes bitte", "user_id": uid}).json()
    assert body["action"]["type"] == "clarify" and body["clarifying_question"]
    assert body["products"] and all(p["category"] == "Sport & Fitness" for p in body["products"][:1])
    # suggestions must not feed back into the profile
    assert body["user_context"]["interests"] == {"Sport & Fitness": 100.0}


def test_clarify_cold_start_stays_pure(client):
    body = client.post("/assist", json={"query": "irgendwas Schönes bitte",
                                        "user_id": "fresh-cold-user"}).json()
    assert body["action"]["type"] == "clarify" and not body["products"]


# ---------- user context (interest profile) ----------

def test_context_percentages():
    from app import context
    context.record("u1", ["Baby & Kind", "Baby & Kind", "Süßigkeiten & Snacks"])
    ints = context.interests("u1")
    assert ints["Baby & Kind"] == 66.7 and ints["Süßigkeiten & Snacks"] == 33.3
    assert context.prompt_context("u1").startswith("Baby & Kind 66.7%")
    assert context.prompt_context("nobody") is None


def test_context_accumulates_over_requests(client):
    r1 = client.post("/assist", json={"query": "Windeln", "user_id": "ctx-user"}).json()
    assert r1["user_context"]["interests"].get("Baby & Kind") == 100.0
    r2 = client.post("/assist", json={"query": "Schokolade", "user_id": "ctx-user"}).json()
    ints = r2["user_context"]["interests"]
    assert "Baby & Kind" in ints and "Süßigkeiten & Snacks" in ints
    assert abs(sum(ints.values()) - 100.0) < 1.0


def test_context_passed_to_llm(client, monkeypatch):
    from app import llm
    captured = {}

    def fake_classify(query, user_context=None, model=None):
        captured["ctx"] = user_context
        return {"intent": "search", "language": "de", "partner": None,
                "search_terms": "windeln", "clarifying_question": None}

    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "backend", lambda: "test")
    monkeypatch.setattr(llm, "model_name", lambda override=None: "fake")
    monkeypatch.setattr(llm, "classify", fake_classify)
    ask_with = lambda q: client.post("/assist", json={"query": q, "user_id": "llm-ctx"}).json()
    ask_with("Windeln und Schnuller")            # builds profile via rules
    ask_with("brauche was fuer den kleinen wonneproppen")  # unknown tokens -> LLM
    assert captured["ctx"] and "Baby & Kind" in captured["ctx"]


def test_llm_escalation_on_unknown_tokens(client, monkeypatch):
    from app import llm
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "backend", lambda: "test")
    monkeypatch.setattr(llm, "model_name", lambda override=None: "fake-model")
    monkeypatch.setattr(llm, "classify", lambda q, user_context=None, model=None: {
        "intent": "search", "language": "de", "partner": None,
        "search_terms": "eier frühstück", "clarifying_question": None,
    })
    body = ask(client, "spiegelei fur fruhstuck")
    assert body["engine"] == "classifier+fake-model@test"
    names = " ".join(p["name"] for p in body["products"])
    assert "Eier" in names
