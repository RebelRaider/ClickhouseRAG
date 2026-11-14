import pytest
from clickhouse_connect.driver.client import Client

from clickhouserag import Chunk, ClickHouseStore, EmbeddedChunk, GraphStore, Relation, SchemaError

pytestmark = pytest.mark.integration


@pytest.fixture
def graph(store, database_client, table):
    graph = GraphStore(store)
    graph.initialize()
    try:
        yield graph
    finally:
        database_client.command(
            "DROP TABLE IF EXISTS {table:Identifier}", {"table": f"{table}_relations"}
        )


def put(
    store: ClickHouseStore,
    chunk_id: str,
    *,
    namespace: str = "default",
    revision: int = 1,
    team: str = "ops",
) -> None:
    store.insert(
        [
            EmbeddedChunk(
                Chunk(chunk_id, "runbook", "A fact", f"ref:{chunk_id}", revision, {"team": team}),
                [1, 0],
            )
        ],
        namespace=namespace,
    )


def test_cited_multi_hop_and_cycle(graph: GraphStore, store: ClickHouseStore) -> None:
    for chunk_id in ("a", "b", "c", "other"):
        put(store, chunk_id)
    put(store, "a", namespace="other")
    facts = [
        Relation("Gateway", "calls", "Ledger", "a"),
        Relation("Ledger", "writes", "Store", "b"),
        Relation("Store", "calls", "Gateway", "c"),
    ]
    assert graph.insert([*facts, facts[0]]) == 3
    graph.insert(facts)
    graph.insert([Relation("Gateway", "calls", "Private", "a")], namespace="other")

    paths = graph.expand([" GATEWAY "], depth=4, metadata={"team": "ops"})
    assert [path.target for path in paths] == ["ledger", "store"]
    assert [step.chunk.source for step in paths[-1].steps] == ["ref:a", "ref:b"]
    assert graph.seed_entities("Who calls the GATEWAY?") == ["gateway"]
    typed_paths = graph.expand(["Gateway"], predicates=["calls", "writes"])
    assert [path.target for path in typed_paths] == ["store"]


def test_revision_and_scope_apply_before_traversal(
    graph: GraphStore, store: ClickHouseStore, database_client: Client, table: str
) -> None:
    database_client.command("SYSTEM STOP MERGES {table:Identifier}", {"table": table})
    put(store, "a")
    put(store, "b", team="finance")
    put(store, "c")
    graph.insert(
        [
            Relation("Alpha", "owns", "Old", "a"),
            Relation("Alpha", "owns", "Zeta", "a", 2),
            Relation("Zeta", "uses", "Hidden", "b"),
            Relation("Hidden", "owns", "Public", "c"),
        ]
    )
    assert [path.target for path in graph.expand(["Alpha"])] == ["old"]
    put(store, "a", revision=2)
    assert [path.target for path in graph.expand(["Alpha"], metadata={"team": "ops"})] == ["zeta"]
    assert graph.seed_entities("Find Old") == []
    assert graph.seed_entities("Find Hidden", metadata={"team": "ops"}) == ["hidden"]
    assert graph.seed_entities("Find Zeta", metadata={"team": "ops"}) == ["zeta"]


def test_fanout_limit_and_empty(graph: GraphStore, store: ClickHouseStore) -> None:
    put(store, "base")
    graph.insert([Relation("A", "rel", item, "base") for item in ["z", "y", "x"]])
    assert [x.target for x in graph.expand(["a"], fanout=2, limit=1)] == ["x"]
    assert graph.expand([]) == []
    with pytest.raises(ValueError, match="2 labels"):
        graph.expand(["a"], predicates=["rel"])


def test_refuses_unrelated_table(store: ClickHouseStore, table: str) -> None:
    graph = GraphStore(store, table=table)
    with pytest.raises(SchemaError, match="graph table"):
        graph.initialize()


def test_knowledge_solves_only_a_complete_and_well_typed_chain(
    graph: GraphStore, store: ClickHouseStore
) -> None:
    from clickhouserag import Knowledge, KnowledgeQuery, Predicate

    put(store, "a")
    put(store, "b")
    put(store, "c", team="private")
    graph.insert(
        [
            Relation(
                "Gateway", "calls", "Ledger", "a", subject_type="service", object_type="service"
            ),
            Relation("Ledger", "writes", "Store", "b", subject_type="service", object_type="db"),
            Relation(
                "Ledger", "writes", "Wrong", "b", subject_type="service", object_type="person"
            ),
            Relation("Ledger", "writes", "Untyped", "b"),
            Relation("Ledger", "writes", "Private", "c", subject_type="service", object_type="db"),
        ]
    )
    kg = Knowledge(
        graph, (Predicate("calls", "service", "service"), Predicate("writes", "service", "db"))
    )
    plan = KnowledgeQuery("GATEWAY", "service", ("calls", "writes"))
    result = kg.solve(plan, metadata={"team": "ops"})
    assert result.entities == ("store",)
    assert [step.chunk.chunk_id for step in result.evidence[0].steps] == ["a", "b"]
    assert kg.solve(KnowledgeQuery("Ledger", "service", ("calls", "writes"))).entities == ()
    with pytest.raises(ValueError, match="does not accept type"):
        kg.solve(KnowledgeQuery("Ledger", "service", ("writes", "calls")))
    with pytest.raises(ValueError, match="unknown predicate"):
        kg.solve(KnowledgeQuery("Ledger", "service", ("unknown",)))
