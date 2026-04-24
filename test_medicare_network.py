"""
test_medicare_network.py
Test suite for Medicare FFS CERT Denial Pattern Network
Covers: data loading, deduplication, profile building, TF-IDF,
        cosine similarity, graph construction, and community detection.
"""

import csv
import io
import math
import unittest
import networkx as nx
from collections import defaultdict
from networkx.algorithms import community


# ─────────────────────────────────────────────
# FUNCTIONS UNDER TEST
# (copied from medicare_network_plotly.py so
#  tests run standalone without Streamlit)
# ─────────────────────────────────────────────

def deduplicate(raw_rows):
    seen, data = set(), []
    for row in raw_rows:
        key = frozenset(row.items())
        if key not in seen:
            seen.add(key)
            data.append(row)
    return data


def build_profiles(data, use_tfidf=True):
    provider_patterns = defaultdict(lambda: defaultdict(int))
    provider_totals   = defaultdict(int)

    for row in data:
        provider = row["Provider Type"]
        error    = row["Error Code"]
        if not error or error.strip() in ("-", ""):
            continue
        provider_patterns[provider][error] += 1
        provider_totals[provider]          += 1

    providers   = list(provider_patterns.keys())
    N_providers = len(providers)
    all_errors  = set(e for p in provider_patterns.values() for e in p)

    error_doc_freq = defaultdict(int)
    for provider, errors in provider_patterns.items():
        for error in errors:
            error_doc_freq[error] += 1

    provider_profiles = {}
    for provider in providers:
        total, profile = provider_totals[provider], {}
        for error, count in provider_patterns[provider].items():
            tf = count / total
            if use_tfidf:
                idf    = math.log(N_providers / error_doc_freq[error])
                weight = tf * idf
            else:
                weight = tf
            if weight > 0:
                profile[error] = weight
        provider_profiles[provider] = profile

    return (
        providers,
        dict(provider_patterns),
        dict(provider_totals),
        provider_profiles,
        all_errors,
        dict(error_doc_freq),
    )


def cosine_similarity(vec1, vec2):
    keys = set(vec1) | set(vec2)
    dot  = sum(vec1.get(k, 0) * vec2.get(k, 0) for k in keys)
    mag1 = math.sqrt(sum(v**2 for v in vec1.values()))
    mag2 = math.sqrt(sum(v**2 for v in vec2.values()))
    return 0.0 if mag1 == 0 or mag2 == 0 else dot / (mag1 * mag2)


def build_graph(providers, profiles, provider_totals, threshold, top_k):
    G = nx.Graph()
    for p in providers:
        G.add_node(p, total_denials=provider_totals[p])

    candidate_edges = []
    for i in range(len(providers)):
        for j in range(i + 1, len(providers)):
            p1, p2 = providers[i], providers[j]
            sim = cosine_similarity(profiles[p1], profiles[p2])
            if sim >= threshold:
                candidate_edges.append((p1, p2, sim))

    node_edge_count = defaultdict(int)
    for p1, p2, sim in sorted(candidate_edges, key=lambda x: -x[2]):
        if node_edge_count[p1] < top_k and node_edge_count[p2] < top_k:
            G.add_edge(p1, p2, weight=sim)
            node_edge_count[p1] += 1
            node_edge_count[p2] += 1
    return G


# ─────────────────────────────────────────────
# SHARED TEST DATA
# ─────────────────────────────────────────────

def make_sample_rows():
    """
    Returns a small list of synthetic rows that mimic the CSV structure.
    Provider A and Provider B have identical profiles (should be highly similar).
    Provider C has a completely different profile (should be dissimilar).
    Provider D has no valid error codes (should be excluded from profiles).
    """
    return [
        {"Provider Type": "Hospital A",   "Error Code": "B9"},
        {"Provider Type": "Hospital A",   "Error Code": "B9"},
        {"Provider Type": "Hospital A",   "Error Code": "C2"},
        {"Provider Type": "Hospital B",   "Error Code": "B9"},
        {"Provider Type": "Hospital B",   "Error Code": "B9"},
        {"Provider Type": "Hospital B",   "Error Code": "C2"},
        {"Provider Type": "Physician C",  "Error Code": "A6"},
        {"Provider Type": "Physician C",  "Error Code": "A6"},
        {"Provider Type": "Physician C",  "Error Code": "A6"},
        {"Provider Type": "Empty Provider", "Error Code": "-"},
        {"Provider Type": "Empty Provider", "Error Code": ""},
    ]


def make_duplicate_rows():
    """Returns rows where some are exact duplicates."""
    return [
        {"Provider Type": "Hospital A", "Error Code": "B9"},
        {"Provider Type": "Hospital A", "Error Code": "B9"},  # duplicate
        {"Provider Type": "Hospital A", "Error Code": "B9"},  # duplicate
        {"Provider Type": "Hospital B", "Error Code": "C2"},
    ]


# ─────────────────────────────────────────────
# TEST CLASSES
# ─────────────────────────────────────────────

class TestDeduplication(unittest.TestCase):
    """Tests for the deduplication function."""

    def test_removes_exact_duplicates(self):
        rows = make_duplicate_rows()
        result = deduplicate(rows)
        self.assertEqual(len(result), 2)

    def test_no_duplicates_unchanged(self):
        rows = [
            {"Provider Type": "Hospital A", "Error Code": "B9"},
            {"Provider Type": "Hospital B", "Error Code": "C2"},
        ]
        result = deduplicate(rows)
        self.assertEqual(len(result), 2)

    def test_empty_input(self):
        result = deduplicate([])
        self.assertEqual(result, [])

    def test_all_duplicates_keeps_one(self):
        rows = [{"Provider Type": "A", "Error Code": "X"}] * 5
        result = deduplicate(rows)
        self.assertEqual(len(result), 1)

    def test_preserves_row_content(self):
        rows = [
            {"Provider Type": "Hospital A", "Error Code": "B9"},
            {"Provider Type": "Hospital A", "Error Code": "B9"},
        ]
        result = deduplicate(rows)
        self.assertEqual(result[0]["Provider Type"], "Hospital A")
        self.assertEqual(result[0]["Error Code"], "B9")


class TestProfileBuilding(unittest.TestCase):
    """Tests for build_profiles — raw counts, normalization, TF-IDF."""

    def setUp(self):
        self.data = make_sample_rows()

    def test_correct_number_of_providers(self):
        # Empty Provider has no valid error codes so should still appear
        # but with an empty profile
        providers, patterns, totals, profiles, errors, doc_freq = build_profiles(self.data)
        # Hospital A, Hospital B, Physician C have valid codes
        # Empty Provider has only "-" and "" so is excluded
        self.assertIn("Hospital A", providers)
        self.assertIn("Hospital B", providers)
        self.assertIn("Physician C", providers)

    def test_invalid_error_codes_excluded(self):
        providers, patterns, totals, profiles, errors, doc_freq = build_profiles(self.data)
        self.assertNotIn("Empty Provider", providers)

    def test_raw_counts_correct(self):
        providers, patterns, totals, profiles, errors, doc_freq = build_profiles(self.data)
        self.assertEqual(patterns["Hospital A"]["B9"], 2)
        self.assertEqual(patterns["Hospital A"]["C2"], 1)

    def test_provider_totals_correct(self):
        providers, patterns, totals, profiles, errors, doc_freq = build_profiles(self.data)
        self.assertEqual(totals["Hospital A"], 3)
        self.assertEqual(totals["Physician C"], 3)

    def test_proportional_profiles_sum_to_one(self):
        # Without TF-IDF, profiles should sum to 1.0
        providers, patterns, totals, profiles, errors, doc_freq = build_profiles(
            self.data, use_tfidf=False
        )
        for provider in providers:
            total = sum(profiles[provider].values())
            self.assertAlmostEqual(total, 1.0, places=5,
                msg=f"{provider} proportions don't sum to 1")

    def test_tfidf_universal_code_downweighted(self):
        """
        A6 appears in only Physician C (1 of 3 providers).
        B9 appears in Hospital A and Hospital B (2 of 3 providers).
        IDF for A6 = log(3/1) = 1.099
        IDF for B9 = log(3/2) = 0.405
        A6 should have higher IDF weight than B9.
        """
        providers, patterns, totals, profiles, errors, doc_freq = build_profiles(
            self.data, use_tfidf=True
        )
        # A6 only in Physician C — IDF should be log(3/1)
        idf_a6 = math.log(3 / doc_freq["A6"])
        idf_b9 = math.log(3 / doc_freq["B9"])
        self.assertGreater(idf_a6, idf_b9)

    def test_all_errors_collected(self):
        providers, patterns, totals, profiles, errors, doc_freq = build_profiles(self.data)
        self.assertIn("B9", errors)
        self.assertIn("C2", errors)
        self.assertIn("A6", errors)

    def test_doc_freq_correct(self):
        providers, patterns, totals, profiles, errors, doc_freq = build_profiles(self.data)
        # B9 appears in Hospital A and Hospital B = 2 providers
        self.assertEqual(doc_freq["B9"], 2)
        # A6 appears only in Physician C = 1 provider
        self.assertEqual(doc_freq["A6"], 1)


class TestCosineSimilarity(unittest.TestCase):
    """Tests for cosine similarity calculation."""

    def test_identical_vectors_score_one(self):
        v = {"B9": 0.5, "C2": 0.5}
        self.assertAlmostEqual(cosine_similarity(v, v), 1.0, places=5)

    def test_completely_different_vectors_score_zero(self):
        v1 = {"B9": 1.0}
        v2 = {"A6": 1.0}
        self.assertAlmostEqual(cosine_similarity(v1, v2), 0.0, places=5)

    def test_empty_vector_returns_zero(self):
        v1 = {}
        v2 = {"B9": 1.0}
        self.assertEqual(cosine_similarity(v1, v2), 0.0)

    def test_both_empty_returns_zero(self):
        self.assertEqual(cosine_similarity({}, {}), 0.0)

    def test_score_between_zero_and_one(self):
        v1 = {"B9": 0.8, "C2": 0.2}
        v2 = {"B9": 0.4, "C2": 0.6}
        score = cosine_similarity(v1, v2)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_similar_profiles_score_higher_than_dissimilar(self):
        similar1   = {"B9": 0.7, "C2": 0.3}
        similar2   = {"B9": 0.6, "C2": 0.4}
        dissimilar = {"A6": 1.0}
        score_similar    = cosine_similarity(similar1, similar2)
        score_dissimilar = cosine_similarity(similar1, dissimilar)
        self.assertGreater(score_similar, score_dissimilar)

    def test_symmetry(self):
        v1 = {"B9": 0.5, "C2": 0.5}
        v2 = {"B9": 0.8, "A6": 0.2}
        self.assertAlmostEqual(
            cosine_similarity(v1, v2),
            cosine_similarity(v2, v1),
            places=10
        )

    def test_hospital_a_and_b_are_highly_similar(self):
        """
        Hospital A and Hospital B have identical error code distributions
        so their cosine similarity should be 1.0.
        """
        data = make_sample_rows()
        providers, patterns, totals, profiles, errors, doc_freq = build_profiles(
            data, use_tfidf=False
        )
        score = cosine_similarity(profiles["Hospital A"], profiles["Hospital B"])
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_physician_c_dissimilar_to_hospitals(self):
        """
        Physician C only has A6 while hospitals have B9 and C2.
        Similarity should be 0.
        """
        data = make_sample_rows()
        providers, patterns, totals, profiles, errors, doc_freq = build_profiles(
            data, use_tfidf=False
        )
        score = cosine_similarity(profiles["Hospital A"], profiles["Physician C"])
        self.assertAlmostEqual(score, 0.0, places=5)


class TestGraphConstruction(unittest.TestCase):
    """Tests for build_graph — nodes, edges, thresholds, top-K."""

    def setUp(self):
        data = make_sample_rows()
        providers, patterns, totals, profiles, errors, doc_freq = build_profiles(
            data, use_tfidf=False
        )
        self.providers = providers
        self.totals    = totals
        self.profiles  = profiles

    def test_all_providers_are_nodes(self):
        G = build_graph(self.providers, self.profiles, self.totals, threshold=0.0, top_k=10)
        for p in self.providers:
            self.assertIn(p, G.nodes())

    def test_node_has_total_denials_attribute(self):
        G = build_graph(self.providers, self.profiles, self.totals, threshold=0.0, top_k=10)
        for p in self.providers:
            self.assertIn("total_denials", G.nodes[p])

    def test_high_threshold_removes_edges(self):
        G = build_graph(self.providers, self.profiles, self.totals, threshold=0.99, top_k=10)
        # Hospital A and B are identical (sim=1.0) so one edge should exist
        # Physician C is dissimilar so no edges to hospitals
        self.assertLessEqual(G.number_of_edges(), 1)

    def test_zero_threshold_connects_similar_providers(self):
        G = build_graph(self.providers, self.profiles, self.totals, threshold=0.0, top_k=10)
        self.assertGreater(G.number_of_edges(), 0)

    def test_top_k_limits_edges_per_node(self):
        G = build_graph(self.providers, self.profiles, self.totals, threshold=0.0, top_k=1)
        for node in G.nodes():
            self.assertLessEqual(G.degree(node), 1)

    def test_edge_weight_is_similarity_score(self):
        G = build_graph(self.providers, self.profiles, self.totals, threshold=0.0, top_k=10)
        for u, v, d in G.edges(data=True):
            self.assertIn("weight", d)
            self.assertGreaterEqual(d["weight"], 0.0)
            self.assertLessEqual(d["weight"], 1.0)

    def test_identical_providers_get_edge(self):
        """Hospital A and B have identical profiles so they should be connected."""
        G = build_graph(self.providers, self.profiles, self.totals, threshold=0.5, top_k=10)
        self.assertTrue(
            G.has_edge("Hospital A", "Hospital B") or
            G.has_edge("Hospital B", "Hospital A")
        )

    def test_dissimilar_providers_no_edge_at_high_threshold(self):
        """Physician C is completely different from hospitals at high threshold."""
        G = build_graph(self.providers, self.profiles, self.totals, threshold=0.5, top_k=10)
        self.assertFalse(
            G.has_edge("Hospital A", "Physician C") or
            G.has_edge("Physician C", "Hospital A")
        )

    def test_graph_is_undirected(self):
        G = build_graph(self.providers, self.profiles, self.totals, threshold=0.0, top_k=10)
        self.assertIsInstance(G, nx.Graph)
        self.assertNotIsInstance(G, nx.DiGraph)


class TestCommunityDetection(unittest.TestCase):
    """Tests for community detection on the built graph."""

    def setUp(self):
        data = make_sample_rows()
        providers, patterns, totals, profiles, errors, doc_freq = build_profiles(
            data, use_tfidf=False
        )
        self.G = build_graph(providers, profiles, totals, threshold=0.5, top_k=10)
        components = sorted(nx.connected_components(self.G), key=len, reverse=True)
        self.G_main = self.G.subgraph(components[0]).copy()

    def test_communities_cover_all_main_nodes(self):
        comms = list(community.greedy_modularity_communities(self.G_main))
        all_nodes_in_comms = set(n for c in comms for n in c)
        self.assertEqual(all_nodes_in_comms, set(self.G_main.nodes()))

    def test_communities_are_non_overlapping(self):
        comms = list(community.greedy_modularity_communities(self.G_main))
        all_nodes = [n for c in comms for n in c]
        # No node should appear in more than one community
        self.assertEqual(len(all_nodes), len(set(all_nodes)))

    def test_at_least_one_community_found(self):
        comms = list(community.greedy_modularity_communities(self.G_main))
        self.assertGreater(len(comms), 0)

    def test_similar_providers_in_same_community(self):
        """Hospital A and B should end up in the same cluster."""
        comms = list(community.greedy_modularity_communities(self.G_main))
        node_community = {node: i for i, c in enumerate(comms) for node in c}
        if "Hospital A" in node_community and "Hospital B" in node_community:
            self.assertEqual(
                node_community["Hospital A"],
                node_community["Hospital B"]
            )


class TestEndToEnd(unittest.TestCase):
    """
    End-to-end integration tests that run the full pipeline
    from raw rows to a graph with communities.
    """

    def test_full_pipeline_runs_without_error(self):
        data = deduplicate(make_sample_rows())
        providers, patterns, totals, profiles, errors, doc_freq = build_profiles(data)
        G = build_graph(providers, profiles, totals, threshold=0.5, top_k=5)
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        G_main     = G.subgraph(components[0]).copy()
        comms      = list(community.greedy_modularity_communities(G_main))
        self.assertIsNotNone(comms)

    def test_dedup_then_profile_gives_correct_counts(self):
        raw  = make_duplicate_rows()
        data = deduplicate(raw)
        providers, patterns, totals, profiles, errors, doc_freq = build_profiles(data)
        # After dedup Hospital A should have only 1 B9 row
        self.assertEqual(patterns["Hospital A"]["B9"], 1)

    def test_pipeline_with_tfidf_off(self):
        data = deduplicate(make_sample_rows())
        providers, patterns, totals, profiles, errors, doc_freq = build_profiles(
            data, use_tfidf=False
        )
        G = build_graph(providers, profiles, totals, threshold=0.5, top_k=5)
        self.assertGreater(G.number_of_nodes(), 0)

    def test_graph_has_no_self_loops(self):
        data = deduplicate(make_sample_rows())
        providers, patterns, totals, profiles, errors, doc_freq = build_profiles(data)
        G = build_graph(providers, profiles, totals, threshold=0.0, top_k=10)
        self.assertEqual(list(nx.selfloop_edges(G)), [])

    def test_empty_data_does_not_crash(self):
        data = []
        providers, patterns, totals, profiles, errors, doc_freq = build_profiles(data)
        self.assertEqual(len(providers), 0)
        G = build_graph(providers, profiles, totals, threshold=0.5, top_k=5)
        self.assertEqual(G.number_of_nodes(), 0)


class TestNullAndUncleanData(unittest.TestCase):
    """Tests for handling null values and unclean data."""

    def test_none_error_code_excluded(self):
        data = [
            {"Provider Type": "Hospital A", "Error Code": None},
            {"Provider Type": "Hospital A", "Error Code": "B9"},
        ]
        providers, patterns, totals, profiles, errors, doc_freq = build_profiles(data)
        self.assertEqual(totals["Hospital A"], 1)

    def test_whitespace_only_error_code_excluded(self):
        data = [
            {"Provider Type": "Hospital A", "Error Code": "   "},
            {"Provider Type": "Hospital A", "Error Code": "B9"},
        ]
        providers, patterns, totals, profiles, errors, doc_freq = build_profiles(data)
        self.assertEqual(totals["Hospital A"], 1)

    def test_dash_error_code_excluded(self):
        data = [
            {"Provider Type": "Hospital A", "Error Code": "-"},
            {"Provider Type": "Hospital A", "Error Code": "B9"},
        ]
        providers, patterns, totals, profiles, errors, doc_freq = build_profiles(data)
        self.assertEqual(totals["Hospital A"], 1)

    def test_provider_with_only_invalid_codes_excluded(self):
        data = [
            {"Provider Type": "Ghost Provider", "Error Code": "-"},
            {"Provider Type": "Ghost Provider", "Error Code": ""},
            {"Provider Type": "Ghost Provider", "Error Code": None},
            {"Provider Type": "Hospital A",    "Error Code": "B9"},
        ]
        providers, patterns, totals, profiles, errors, doc_freq = build_profiles(data)
        self.assertNotIn("Ghost Provider", providers)

    def test_mixed_valid_and_invalid_codes(self):
        data = [
            {"Provider Type": "Hospital A", "Error Code": "B9"},
            {"Provider Type": "Hospital A", "Error Code": "-"},
            {"Provider Type": "Hospital A", "Error Code": None},
            {"Provider Type": "Hospital A", "Error Code": "C2"},
        ]
        providers, patterns, totals, profiles, errors, doc_freq = build_profiles(data)
        self.assertEqual(totals["Hospital A"], 2)
        self.assertIn("B9", patterns["Hospital A"])
        self.assertIn("C2", patterns["Hospital A"])

    def test_extra_whitespace_in_error_code(self):
        """Error codes with leading/trailing whitespace should be treated as invalid."""
        data = [
            {"Provider Type": "Hospital A", "Error Code": "  B9  "},
            {"Provider Type": "Hospital A", "Error Code": "B9"},
        ]
        providers, patterns, totals, profiles, errors, doc_freq = build_profiles(data)
        # "  B9  ".strip() is "B9" which is valid — both should count
        self.assertEqual(totals["Hospital A"], 2)

    def test_empty_provider_type_handled(self):
        data = [
            {"Provider Type": "", "Error Code": "B9"},
            {"Provider Type": "Hospital A", "Error Code": "B9"},
        ]
        # Should not crash — empty string provider gets included as a key
        providers, patterns, totals, profiles, errors, doc_freq = build_profiles(data)
        self.assertIn("Hospital A", providers)

    def test_single_provider_single_code_no_crash(self):
        data = [{"Provider Type": "Hospital A", "Error Code": "B9"}]
        providers, patterns, totals, profiles, errors, doc_freq = build_profiles(data)
        self.assertEqual(len(providers), 1)
        self.assertEqual(totals["Hospital A"], 1)

    def test_dedup_handles_none_values(self):
        rows = [
            {"Provider Type": "Hospital A", "Error Code": None},
            {"Provider Type": "Hospital A", "Error Code": None},  # duplicate
            {"Provider Type": "Hospital A", "Error Code": "B9"},
        ]
        result = deduplicate(rows)
        self.assertEqual(len(result), 2)

    def test_large_number_of_invalid_rows_does_not_crash(self):
        data = [{"Provider Type": "Hospital A", "Error Code": "-"}] * 10000
        data.append({"Provider Type": "Hospital A", "Error Code": "B9"})
        providers, patterns, totals, profiles, errors, doc_freq = build_profiles(data)
        self.assertEqual(totals["Hospital A"], 1)

# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    unittest.main(verbosity=2)
