"""Shared swarm memory with per-drone permissions.

Datasets and rules (the same table drives both backends):
  private_<drone>  only that drone reads and writes
  swarm_map        scouts write; everyone reads (landmark observations, hazards)
  outcomes         only the operator writes; everyone reads (mission results, corrections)

Backends:
  cognee  one cognee user per drone plus 'operator'; datasets are owned by their writer of record and
          shared with give_permission_on_dataset; a write without permission is rejected by cognee itself
          (PermissionDeniedError), which is what the UI shows as 'denied'.
  local   used only when cognee is not configured (no SIE key); enforces the same table in Python and is
          labelled 'local fallback' everywhere it is shown.

Every memory is a short sentence followed by a machine-readable tag, e.g.
  "scout-2 felt a strong gust near (6.1, 4.3). [fact kind=hazard hazard=gust x=6.1 y=4.3 r=3.0 by=scout-2 id=f12]"
so the courier's planner can turn recalled text back into avoid-circles without trusting free text.
"""
import asyncio
import itertools
import os
import re
import time
import uuid
from dataclasses import dataclass, field

DRONES = ["scout-1", "scout-2", "courier"]
USERS = ["operator"] + DRONES


def policy():
    p = {"swarm_map": {"read": set(USERS), "write": {"scout-1", "scout-2"}},
         "outcomes": {"read": set(USERS), "write": {"operator"}}}
    for d in DRONES:
        p[f"private_{d}"] = {"read": {d}, "write": {d}}
    return p


POLICY = policy()
OWNER = {"swarm_map": "operator", "outcomes": "operator", **{f"private_{d}": d for d in DRONES}}
TAG_RE = re.compile(r"\[fact ([^\]]+)\]")
_ids = itertools.count(1)


def can(user, dataset, perm):
    return user in POLICY.get(dataset, {}).get(perm, set())


def make_tag(**kv):
    return "[fact " + " ".join(f"{k}={v}" for k, v in kv.items()) + "]"


def parse_tags(text):
    out = []
    for m in TAG_RE.finditer(text):
        d = {}
        for tok in m.group(1).split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                try:
                    v = float(v) if re.fullmatch(r"-?\d+(\.\d+)?", v) else v
                except ValueError:
                    pass
                d[k] = v
        out.append(d)
    return out


@dataclass
class MemEvent:
    kind: str            # write | read | denied | feedback | improve | error
    user: str
    dataset: str
    text: str
    ok: bool = True
    t: float = field(default_factory=time.time)
    meta: dict = field(default_factory=dict)

    def as_dict(self):
        return {"kind": self.kind, "user": self.user, "dataset": self.dataset, "text": self.text, "ok": self.ok,
                "t": self.t, "meta": self.meta}


class LocalBackend:
    name = "local fallback (cognee not configured)"

    def __init__(self):
        self.docs = {ds: [] for ds in POLICY}

    async def setup(self):
        return None

    async def write(self, user, dataset, text, node_set=None):
        if not can(user, dataset, "write"):
            raise PermissionError(f"{user} does not have permission to write to dataset: {dataset}")
        self.docs[dataset].append({"text": text, "user": user, "node_set": node_set or [], "t": time.time()})

    async def recall(self, user, query, datasets, session_id=None):
        out = []
        for ds in datasets:
            if not can(user, ds, "read"):
                raise PermissionError(f"{user} does not have permission to read dataset: {ds}")
            q = set(re.findall(r"[a-z_]+", query.lower()))
            for doc in self.docs[ds]:
                score = len(q & set(re.findall(r"[a-z_]+", doc["text"].lower())))
                out.append((score, ds, doc["text"]))
        out.sort(key=lambda x: -x[0])
        return [{"dataset": ds, "text": t} for _, ds, t in out[:30]]

    async def feedback(self, user, session_id, score, text):
        return False

    async def improve(self, user, dataset, session_ids):
        return "skipped (local fallback has no improve step)"

    def graph(self):
        nodes, edges = [], []
        for ds in ("swarm_map", "outcomes"):
            for i, doc in enumerate(self.docs[ds]):
                nid = f"{ds}:{i}"
                nodes.append({"id": nid, "label": doc["text"][:60], "dataset": ds, "by": doc["user"]})
                for tag in parse_tags(doc["text"]):
                    for key in ("landmark", "hazard"):
                        if key in tag:
                            ent = f"{key}:{tag[key]}"
                            if not any(n["id"] == ent for n in nodes):
                                nodes.append({"id": ent, "label": str(tag[key]), "dataset": "entity", "by": "entity"})
                            edges.append({"source": nid, "target": ent})
        return {"nodes": nodes, "edges": edges}


class CogneeBackend:
    name = "cognee"

    def __init__(self):
        self.users = {}
        self.dataset_ids = {}
        self.texts = {ds: [] for ds in POLICY}      # what was written, for the graph view's contributor colours

    async def setup(self):
        import cognee
        from cognee.modules.engine.operations.setup import setup
        from cognee.modules.users.methods import create_user, get_user_by_email
        from cognee.modules.users.permissions.methods import give_permission_on_dataset
        await setup()
        for u in USERS:
            email = f"{u}@flybrain-swarm.example"
            try:
                user = await get_user_by_email(email)
            except Exception:
                user = None
            if user is None:
                user = await create_user(email=email, password=f"swarm-{u}-pw", is_superuser=False)
            self.users[u] = user
        # create each dataset as its owner with a seed document, then share it per POLICY
        for ds, owner in OWNER.items():
            res = await cognee.remember(f"Dataset {ds} of the Flybrain Swarm, owned by {owner}.", dataset_name=ds,
                                        user=self.users[owner], self_improvement=False)
            self.dataset_ids[ds] = uuid.UUID(str(res.dataset_id))
            for perm in ("read", "write"):
                for u in POLICY[ds][perm]:
                    if u != owner:
                        await give_permission_on_dataset(self.users[u], self.dataset_ids[ds], perm)

    async def write(self, user, dataset, text, node_set=None):
        import cognee
        from cognee.modules.users.exceptions import PermissionDeniedError
        try:
            await cognee.remember(text, dataset_id=self.dataset_ids[dataset], user=self.users[user],
                                  node_set=node_set or None, self_improvement=False)
        except PermissionDeniedError as e:
            raise PermissionError(str(e)) from e
        self.texts[dataset].append({"text": text, "user": user})

    async def recall(self, user, query, datasets, session_id=None):
        import cognee
        from cognee import SearchType
        from cognee.modules.users.exceptions import PermissionDeniedError
        try:
            res = await cognee.recall(query, SearchType.CHUNKS, dataset_ids=[self.dataset_ids[d] for d in datasets],
                                      user=self.users[user], top_k=30, auto_route=False, session_id=session_id,
                                      feedback_influence=0.5)
        except PermissionDeniedError as e:
            raise PermissionError(str(e)) from e
        out = []
        for r in res:
            text = getattr(r, "text", None) or str(r)
            out.append({"dataset": getattr(r, "dataset_name", "") or "", "text": text})
        return out

    async def feedback(self, user, session_id, score, text):
        import cognee
        entries = await cognee.session.get_session(session_id=session_id, last_n=5, user=self.users[user])
        if not entries:
            return False
        return await cognee.session.add_feedback(session_id=session_id, qa_id=entries[-1].qa_id, feedback_text=text,
                                                 feedback_score=int(score), user=self.users[user])

    async def improve(self, user, dataset, session_ids):
        import cognee
        r = await cognee.improve(dataset=self.dataset_ids[dataset], session_ids=session_ids or None, user=self.users[user])
        return getattr(r, "status", str(r))

    def graph(self):
        return LocalBackend.graph(self_as_local(self.texts))


def self_as_local(texts):
    lb = LocalBackend.__new__(LocalBackend)
    lb.docs = texts
    return lb


def cognee_configured():
    return bool(os.environ.get("SIE_API_KEY")) and os.environ.get("MEMORY_BACKEND", "cognee") == "cognee"


class SwarmMemory:
    """Facade used by agents and the API. Records every access as an event for the UI feed."""

    def __init__(self, backend=None):
        self.backend = backend or (CogneeBackend() if cognee_configured() else LocalBackend())
        self.events = []
        self.listeners = []
        self.ready = False

    async def setup(self):
        await self.backend.setup()
        self.ready = True

    def _emit(self, ev):
        self.events.append(ev)
        self.events = self.events[-500:]
        for cb in self.listeners:
            cb(ev)

    async def write(self, user, dataset, text, node_set=None):
        try:
            await self.backend.write(user, dataset, text, node_set)
            self._emit(MemEvent("write", user, dataset, text))
            return True
        except PermissionError as e:
            self._emit(MemEvent("denied", user, dataset, text, ok=False, meta={"reason": str(e)[:200]}))
            return False
        except Exception as e:
            self._emit(MemEvent("error", user, dataset, text, ok=False, meta={"reason": f"{type(e).__name__}: {e}"[:200]}))
            return False

    async def recall(self, user, query, datasets, session_id=None):
        try:
            res = await self.backend.recall(user, query, datasets, session_id)
            self._emit(MemEvent("read", user, ",".join(datasets), query, meta={"hits": len(res)}))
            return res
        except PermissionError as e:
            self._emit(MemEvent("denied", user, ",".join(datasets), query, ok=False, meta={"reason": str(e)[:200]}))
            return []
        except Exception as e:
            self._emit(MemEvent("error", user, ",".join(datasets), query, ok=False, meta={"reason": f"{type(e).__name__}: {e}"[:200]}))
            return []

    async def feedback(self, user, session_id, score, text):
        try:
            ok = await self.backend.feedback(user, session_id, score, text)
        except Exception as e:
            ok = False; text = f"{text} (feedback failed: {type(e).__name__})"
        self._emit(MemEvent("feedback", user, "session", f"score {score}: {text}", ok=bool(ok)))
        return ok

    async def improve(self, user, dataset, session_ids=None):
        try:
            status = await self.backend.improve(user, dataset, session_ids)
            self._emit(MemEvent("improve", user, dataset, f"improve: {status}"))
        except Exception as e:
            self._emit(MemEvent("error", user, dataset, "improve", ok=False, meta={"reason": f"{type(e).__name__}: {e}"[:200]}))

    def graph(self):
        return self.backend.graph()


def new_fact_id():
    return f"f{next(_ids)}"
