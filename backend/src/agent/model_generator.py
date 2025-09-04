import os
import json
import re
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TypedDict

import numpy as np
import requests
from fastapi.exceptions import HTTPException
from pydantic import BaseModel, Field, model_validator
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage
from langgraph.graph import StateGraph, END

from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)


# ----------------- Embedder -----------------


class Embedder:
    """Simple HTTP embedding client."""

    EMBEDDINGS_URL = os.getenv(
        "EMBEDDINGS_URL", "http://87.242.104.103:8080/embeddings"
    )
    EMBEDDINGS_HEADERS = {"Content-Type": "application/json"}
    EMBEDDINGS_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
    TIMEOUT = 15

    @classmethod
    def _find_embedding_in_obj(cls, obj) -> Optional[list]:
        if obj is None:
            return None
        if isinstance(obj, dict):
            if "embedding" in obj and isinstance(obj["embedding"], (list, tuple)):
                return obj["embedding"]
            if "embeddings" in obj and isinstance(obj["embeddings"], (list, tuple)):
                e = obj["embeddings"]
                if e and isinstance(e[0], (list, tuple)):
                    return e[0]
                return e
            if "data" in obj and isinstance(obj["data"], (list, tuple)):
                first = obj["data"][0] if obj["data"] else None
                if first is not None:
                    if isinstance(first, dict) and "embedding" in first:
                        return first["embedding"]
                    if isinstance(first, (list, tuple)):
                        return first
            for v in obj.values():
                found = cls._find_embedding_in_obj(v)
                if found is not None:
                    return found
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                found = cls._find_embedding_in_obj(item)
                if found is not None:
                    return found
        return None

    @classmethod
    def request_to_embed_model(cls, input_query: str) -> Optional[list]:
        data = {"model": cls.EMBEDDINGS_MODEL, "input": input_query}
        try:
            response = requests.post(
                cls.EMBEDDINGS_URL,
                headers=cls.EMBEDDINGS_HEADERS,
                json=data,
                timeout=cls.TIMEOUT,
            )
        except requests.exceptions.RequestException as exc:
            raise HTTPException(status_code=502, detail=f"Embedding request failed: {exc}")

        try:
            j = response.json()
        except ValueError:
            raise HTTPException(
                status_code=502,
                detail=f"Embedding service returned non-JSON response: {response.text[:1000]}",
            )

        if response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail={
                    "message": "Embedding service error",
                    "status_code": response.status_code,
                    "body": j,
                },
            )

        emb = cls._find_embedding_in_obj(j)
        if emb is None:
            raise HTTPException(
                status_code=502,
                detail={"message": "Unable to locate embedding in response JSON", "response_json": j},
            )

        return list(map(float, emb))


# ----------------- Domain models -----------------


@dataclass
class BianElement:
    id: str
    type: str
    name: str
    documentation: str = ""
    properties: Dict[str, str] = field(default_factory=dict)


@dataclass
class BianRelation:
    id: str
    type: str
    source: str
    target: str
    label: str = ""


class DataAttribute(BaseModel):
    name: str
    datatype: str = Field(..., description="string|integer|decimal|date|boolean|uuid|json")
    nullable: bool = True
    derived: bool = False
    description: Optional[str] = None


class Entity(BaseModel):
    name: str
    description: Optional[str] = None
    attributes: List[DataAttribute]
    primary_key: List[str] = Field(default_factory=list)
    source_bian_elements: List[str] = Field(default_factory=list)
    is_extension: bool = False
    extension_reason: Optional[str] = None

    @model_validator(mode="before")
    def normalize_attributes(cls, values):
        if isinstance(values, dict) and isinstance(values.get("attributes"), dict):
            values["attributes"] = [
                {"name": n, "datatype": dt} for n, dt in values["attributes"].items()
            ]
        return values


class Relationship(BaseModel):
    from_entity: str
    to_entity: str
    type: str
    cardinality: str
    description: Optional[str] = None
    source_bian_elements: List[str]


class DataModel(BaseModel):
    product_name: str
    language: str = Field(default="ru")
    entities: List[Entity]
    relationships: List[Relationship]


# ----------------- KB -----------------


class JsonKB:
    def __init__(self, elements_path: str, relations_path: str):
        with open(elements_path, "r", encoding="utf-8") as f:
            raw_elements = json.load(f)
        with open(relations_path, "r", encoding="utf-8") as f:
            raw_relations = json.load(f)
        self.elements: Dict[str, BianElement] = {
            k: BianElement(
                id=v.get("id", k),
                type=v.get("type", "Unknown"),
                name=v.get("name", ""),
                documentation=v.get("documentation", ""),
                properties=v.get("properties", {}) or {},
            )
            for k, v in raw_elements.items()
        }
        self.relations: List[BianRelation] = [
            BianRelation(
                id=r.get("id", ""),
                type=r.get("type", "Association"),
                source=r.get("source"),
                target=r.get("target"),
                label=r.get("label", ""),
            )
            for r in raw_relations
            if r.get("source") in self.elements and r.get("target") in self.elements
        ]
        self.neighbors: Dict[str, List[str]] = {eid: [] for eid in self.elements}
        for rel in self.relations:
            self.neighbors[rel.source].append(rel.target)
            self.neighbors[rel.target].append(rel.source)


# ----------------- Milvus Retriever -----------------


class MilvusRetriever:
    def __init__(self, kb: JsonKB, collection_name: str = "kb_embeddings"):
        self.kb = kb
        self.collection_name = collection_name
        connections.connect(
            "default",
            host=os.getenv("MILVUS_HOST", "milvus"),
            port=os.getenv("MILVUS_PORT", "19530"),
        )
        if not utility.has_collection(collection_name):
            self._build_index()
        self._collection = Collection(collection_name)
        self._collection.load()

    def _build_index(self, batch_size: int = 64):
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="eid", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
        ]
        schema = CollectionSchema(fields)
        collection = Collection(self.collection_name, schema)
        id_map: Dict[int, str] = {}
        ids, texts = [], []
        for eid, el in self.kb.elements.items():
            neigh_names = ", ".join(
                [self.kb.elements[nid].name for nid in self.kb.neighbors.get(eid, [])[:20] if nid in self.kb.elements]
            )
            buff = textwrap.dedent(
                f"""
                ID: {eid}
                Type: {el.type}
                Name: {el.name}
                Documentation: {el.documentation}
                Neighbors: {neigh_names}
                """
            )
            ids.append(eid)
            texts.append(buff)
        pks, e_ids, vectors = [], [], []
        for idx, (eid, txt) in enumerate(zip(ids, texts), 1):
            emb = Embedder.request_to_embed_model(txt)
            if emb is None:
                continue
            e_ids.append(eid)
            vectors.append(emb)
            pks.append(idx)
            id_map[idx] = eid
            if len(pks) >= batch_size:
                collection.insert({"pk": pks, "eid": e_ids, "embedding": vectors})
                pks, e_ids, vectors = [], [], []
        if pks:
            collection.insert({"pk": pks, "eid": e_ids, "embedding": vectors})
        collection.flush()
        collection.create_index(
            "embedding", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 1024}}
        )
        collection.load()
        self.id_map = id_map
        self._collection = collection

    def query(self, request: str, top_k: int) -> List[str]:
        vec = Embedder.request_to_embed_model(request)
        if vec is None:
            return []
        res = self._collection.search(
            [vec],
            "embedding",
            limit=top_k,
            param={"metric_type": "L2", "params": {"nprobe": 10}},
        )
        ctx: List[str] = []
        hit_ids: List[str] = []
        for hit in res[0]:
            eid = self.id_map.get(int(hit.id))
            if not eid:
                continue
            el = self.kb.elements[eid]
            ctx.append(f"[{eid}] {el.type} :: {el.name}\n{el.documentation}\n")
            hit_ids.append(eid)
        for eid in hit_ids:
            for n in self.kb.neighbors.get(eid, [])[:20]:
                if n in self.kb.elements and n not in hit_ids:
                    el = self.kb.elements[n]
                    ctx.append(f"[{n}] {el.type} :: {el.name}\n{el.documentation}\n")
        return ctx


# ----------------- Prompts and util -----------------

SYSTEM_PROMPT = (
    "You are a senior banking data modeler. You MUST align your output to the provided "
    "BIAN context. Do not invent entities or attributes if they are not present or cannot "
    "be justified by BIAN elements. If something is needed but missing, add it with "
    "is_extension=true and a brief extension_reason. Output STRICT JSON matching the schema: "
    "DataModel(product_name, language, entities[], relationships[]). Use Russian field names when appropriate."
)

USER_PROMPT_TEMPLATE = """
Пользовательский запрос (RU):
{request}

Контекст BIAN (релевантные элементы):
{context}

Требования к результату:
1) Верни ТОЛЬКО корректный JSON по схеме: DataModel.
2) Для каждой сущности укажи source_bian_elements (список ID из контекста), primary_key (минимум 1 поле), attributes (тип: string|integer|decimal|date|boolean|uuid|json).
3) Для каждой связи заполни: from_entity, to_entity, type (association|composition|aggregation|identifying|reference), cardinality (1..1|1..*|0..1|0..*), source_bian_elements.
4) Не используй сущности/поля без обоснования в BIAN — пометь их как is_extension=true с кратким extension_reason.
5) product_name = краткое русское имя продукта.
"""


JSON_FENCE = re.compile(r"\{[\s\S]*\}\s*$")


def extract_json_block(text: str) -> str:
    t = text.strip()
    if t.startswith("{") and t.endswith("}"):
        return t
    m = JSON_FENCE.search(t)
    if m:
        return m.group(0)
    return t.replace("```json", "").replace("```", "").strip()


# ----------------- Pipeline Nodes -----------------


class AgentState(TypedDict):
    messages: List
    request: str
    retrieved_context: str
    draft_model: dict
    validated_model: dict
    artifacts: Dict[str, str]


kb = JsonKB("bian-research/out_kb/elements.json", "bian-research/out_kb/relations.json")
retriever = MilvusRetriever(kb)
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    model=os.getenv("CHAT_MODEL_NAME", "qwen2:72b"),
    temperature=0,
)


def node_retrieve(state: AgentState) -> AgentState:
    ctx = retriever.query(state["request"], top_k=int(os.getenv("TOP_K", "20")))
    state["retrieved_context"] = "\n".join(ctx)
    return state


def node_generate(state: AgentState) -> AgentState:
    user_prompt = USER_PROMPT_TEMPLATE.format(
        request=state["request"], context=state["retrieved_context"]
    )
    raw = llm.invoke([
        ("system", SYSTEM_PROMPT),
        ("user", user_prompt),
    ])
    text = raw.content if hasattr(raw, "content") else str(raw)
    data = json.loads(extract_json_block(text))
    state["draft_model"] = data
    return state


def node_validate(state: AgentState) -> AgentState:
    model = DataModel(**state["draft_model"])
    state["validated_model"] = json.loads(model.model_dump_json())
    return state


SQL_TYPE_MAP = {
    "string": "text",
    "integer": "integer",
    "decimal": "numeric",
    "date": "date",
    "boolean": "boolean",
    "uuid": "uuid",
    "json": "jsonb",
}


def render_plantuml(model: DataModel) -> str:
    lines = ["@startuml", "hide circle"]
    alias_map = {}
    for idx, e in enumerate(model.entities, start=1):
        alias = f"E{idx}"
        alias_map[e.name] = alias
        lines.append(f"entity \"{e.name}\" as {alias} {{")
        for a in e.attributes:
            null = "" if not a.nullable else " (nullable)"
            lines.append(f"  {a.name} : {a.datatype}{null}")
        if e.primary_key:
            lines.append("  -- PK --")
            for pk in e.primary_key:
                lines.append(f"  * {pk}")
        lines.append("}")
    for r in model.relationships:
        arrow = {
            "association": "--",
            "reference": "..",
            "composition": "*--",
            "aggregation": "o--",
            "identifying": "+--",
        }.get(r.type, "--")
        lines.append(
            f"{alias_map.get(r.from_entity, r.from_entity)} {arrow} {alias_map.get(r.to_entity, r.to_entity)} : {r.cardinality}"
        )
    lines.append("@enduml")
    return "\n".join(lines)


def sql_ident(name: str) -> str:
    safe = re.sub(r"[^A-Za-zА-Яа-я0-9_]", "_", name)
    return '"' + safe + '"'


def render_sql(model: DataModel) -> str:
    ddl = []
    for e in model.entities:
        cols = []
        for a in e.attributes:
            sql_type = SQL_TYPE_MAP.get(a.datatype.lower(), "text")
            null = " NOT NULL" if not a.nullable else ""
            cols.append(f"\t{sql_ident(a.name)} {sql_type}{null}")
        pk = ""
        if e.primary_key:
            pk = f",\n\tPRIMARY KEY ({', '.join(sql_ident(x) for x in e.primary_key)})"
        ddl.append(
            f"CREATE TABLE {sql_ident(e.name)} (\n" + ",\n".join(cols) + pk + "\n);\n"
        )
    return "\n".join(ddl)


def node_render(state: AgentState) -> AgentState:
    model = DataModel(**state["validated_model"])
    state["artifacts"] = {
        "json": json.dumps(model.model_dump(), ensure_ascii=False, indent=2),
        "puml": render_plantuml(model),
        "sql": render_sql(model),
    }
    state.setdefault("messages", []).append(AIMessage(content=state["artifacts"]["json"]))
    return state


# ----------------- Graph -----------------


def build_graph() -> StateGraph:
    g = StateGraph(AgentState)
    g.add_node("retrieve", node_retrieve)
    g.add_node("generate", node_generate)
    g.add_node("validate", node_validate)
    g.add_node("render", node_render)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "validate")
    g.add_edge("validate", "render")
    g.add_edge("render", END)
    return g.compile()


graph = build_graph()
