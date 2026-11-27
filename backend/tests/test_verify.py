import json

from agentic_rag.agent.verify import Claim, split_claims, unsupported_claims
from tests.fakes import FakeLLM, make_chunk


def test_split_claims_keeps_cited_sentences_and_bullets() -> None:
    answer = (
        "HNSW builds a multilayer graph [1]. It has better query performance than IVFFlat [1][2].\n"
        "- `m` is the max number of connections per layer [3]\n"
        "Here is an example:\n"
        "```sql\nCREATE INDEX ON items USING hnsw (embedding vector_l2_ops) [9];\n```\n"
        "Short [1]."
    )

    assert split_claims(answer) == [
        Claim("HNSW builds a multilayer graph.", (1,)),
        Claim("It has better query performance than IVFFlat.", (1, 2)),
        Claim("`m` is the max number of connections per layer", (3,)),
    ]


async def test_unsupported_claims_checks_each_claim_against_its_own_sources() -> None:
    def reply(messages: list, _schema: object) -> str:
        user = messages[1].content
        return json.dumps({"supported": "graph" in user.split("Claim:")[0]})

    llm = FakeLLM(reply)
    sources = [make_chunk("HNSW creates a graph."), make_chunk("IVFFlat uses lists.")]

    flagged = await unsupported_claims(
        llm, "HNSW creates a layered structure [1]. IVFFlat runs on GPUs [2].", sources
    )

    assert flagged == ["IVFFlat runs on GPUs."]
    assert len(llm.received) == 2


async def test_claim_citing_a_missing_source_is_unsupported_without_a_model_call() -> None:
    llm = FakeLLM('{"supported": true}')

    flagged = await unsupported_claims(llm, "This cites nothing real [7].", [make_chunk()])

    assert flagged == ["This cites nothing real."]
    assert llm.received == []


async def test_verifier_sees_each_source_with_its_title_and_heading() -> None:
    llm = FakeLLM('{"supported": true}')
    source = make_chunk(
        "We use BART-large.", title="RAG paper", page=3, section="2.3 Generator: BART"
    )

    await unsupported_claims(llm, "The generator is BART-large [1].", [source])

    assert (
        "RAG paper > page 3 · 2.3 Generator: BART\nWe use BART-large." in llm.received[0][1].content
    )
