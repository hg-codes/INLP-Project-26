# ==========================================
# 1. Imports & Setup
# ==========================================
!pip install pytorch-crf

import csv, dataclasses, gc, json, logging, os, random, re, sys, time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchcrf import CRF
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"GPUs Available: {num_gpus}")
    for i in range(num_gpus):
        print(f"  GPU {i} : {torch.cuda.get_device_name(i)}")
        print(f"  VRAM {i}: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")


# ==========================================
# 2. Sentence Splitter
# ==========================================
@dataclass
class Sentence:
    text: str
    start_char: int
    end_char: int
    is_header: bool = False
    relative_position: float = 0.0
    follows_header: bool = False
    indent_level: int = 0

_CLAUSE_PATS = [re.compile(p, re.IGNORECASE) for p in [
    r"^\s*\d+\.\s", r"^\s*\d+\.\d+\s", r"^\s*\d+\.\d+\.\d+\s",
    r"^\s*[A-Z]\.\s", r"^\s*\([a-z]\)\s", r"^\s*\([ivx]+\)\s",
    r"^\s*Article\s+\d+", r"^\s*Section\s+\d+",
    r"^\s*Schedule\s+", r"^\s*Exhibit\s+", r"^\s*Annex\s+",
    r"^\s*WHEREAS", r"^\s*NOW,?\s+THEREFORE",
]]
_HDR_PAT    = re.compile(r"^[A-Z\s\d\.\:]{5,80}$")
_INDENT_PAT = re.compile(r"^(\s+)")

def _indent_level(line):
    m = _INDENT_PAT.match(line)
    return len(m.group(1).expandtabs(4)) // 4 if m else 0

def _is_clause_start(line):
    s = line.strip()
    return bool(s) and any(p.match(s) for p in _CLAUSE_PATS)

def _is_header(line):
    s = line.strip()
    return bool(s) and bool(_HDR_PAT.match(s)) and len(s) > 4

def split_contract(text):
    lines, offset = [], 0
    for line in text.splitlines(keepends=True):
        lines.append((line, offset)); offset += len(line)

    merged, buf, buf_start = [], "", 0
    for line_text, off in lines:
        s = line_text.strip()
        if not s:
            if buf.strip(): merged.append((buf, buf_start))
            buf, buf_start = "", off; continue
        if not buf:
            buf, buf_start = line_text, off
        elif _is_clause_start(s) or _is_header(s):
            if buf.strip(): merged.append((buf, buf_start))
            buf, buf_start = line_text, off
        else:
            buf = buf.rstrip("\n") + " " + s + "\n"
    if buf.strip(): merged.append((buf, buf_start))

    sentences, doc_len, prev_hdr = [], max(len(text), 1), False
    for seg, start in merged:
        s = seg.strip()
        if not s: prev_hdr = False; continue
        is_hdr = _is_header(s)
        sentences.append(Sentence(
            text=s, start_char=start, end_char=start + len(seg),
            is_header=is_hdr, relative_position=start / doc_len,
            follows_header=prev_hdr, indent_level=_indent_level(seg),
        ))
        prev_hdr = is_hdr
    return sentences


# ==========================================
# 3. Dataset & Tokenisation
# ==========================================
@dataclass
class RawContract:
    doc_id: str
    text: str
    clauses: List[Tuple[int, int]]

@dataclass
class TokenisedDocument:
    doc_id: str
    input_ids: List[torch.Tensor]
    attention_masks: List[torch.Tensor]
    positional_features: torch.Tensor
    labels: torch.Tensor
    num_sentences: int

def _pos_features(sentences):
    return torch.tensor([
        [s.relative_position, float(s.is_header),
         float(s.follows_header), min(s.indent_level / 4.0, 1.0)]
        for s in sentences], dtype=torch.float)

def _bio_labels(sentences, clauses):
    starts = {s for s, _ in clauses}
    return [int(any(s.start_char <= cs < s.end_char for cs in starts))
            for s in sentences]

def tokenise_contract(raw, tokenizer, max_sent_len=64):
    sents  = split_contract(raw.text)
    labels = _bio_labels(sents, raw.clauses)
    ids_list, mask_list = [], []
    for s in sents:
        enc = tokenizer(s.text, max_length=max_sent_len, padding="max_length",
                        truncation=True, return_tensors="pt")
        ids_list.append(enc["input_ids"].squeeze(0))
        mask_list.append(enc["attention_mask"].squeeze(0))
    return TokenisedDocument(
        doc_id=raw.doc_id, input_ids=ids_list, attention_masks=mask_list,
        positional_features=_pos_features(sents),
        labels=torch.tensor(labels, dtype=torch.long),
        num_sentences=len(sents))

def _truncate(doc, n):
    return TokenisedDocument(
        doc_id=doc.doc_id, input_ids=doc.input_ids[:n],
        attention_masks=doc.attention_masks[:n],
        positional_features=doc.positional_features[:n],
        labels=doc.labels[:n], num_sentences=n)

def _sanitize(text):
    # strip ASCII control chars (null bytes etc.) that break single-line JSON
    return "".join(
        ch if (ord(ch) >= 32 or ord(ch) in (9, 10, 13)) else " "
        for ch in text
    )

class ContractDataset(Dataset):
    def __init__(self, docs, max_sents=512):
        self.docs, self.max_sents = docs, max_sents
    def __len__(self): return len(self.docs)
    def __getitem__(self, i): return self.docs[i]

    @classmethod
    def from_jsonl(cls, path, tokenizer, max_sent_len=64, max_sents=512):
        docs, skipped = [], 0
        raw_bytes = Path(path).read_bytes().decode("utf-8", errors="replace")
        lines = raw_bytes.splitlines()
        for line in tqdm(lines, desc=f"Tokenising {Path(path).name}", leave=False):
            if not line.strip(): continue
            try:
                obj = json.loads(line)
                raw = RawContract(obj["doc_id"], obj["text"],
                                  [tuple(c) for c in obj["clauses"]])
                tok = tokenise_contract(raw, tokenizer, max_sent_len)
                if tok.num_sentences > max_sents: tok = _truncate(tok, max_sents)
                docs.append(tok)
            except Exception as e:
                skipped += 1
        if skipped: print(f"  {skipped} lines skipped")
        return cls(docs, max_sents)

def collate_contracts(batch):
    max_n   = max(d.num_sentences for d in batch)
    seq_len = batch[0].input_ids[0].shape[0]
    B = len(batch)
    input_ids = torch.zeros(B, max_n, seq_len, dtype=torch.long)
    attn_mask = torch.zeros(B, max_n, seq_len, dtype=torch.long)
    sent_mask = torch.zeros(B, max_n, dtype=torch.bool)
    pos_feats = torch.zeros(B, max_n, 4, dtype=torch.float)
    labels    = torch.full((B, max_n), -100, dtype=torch.long)
    for i, doc in enumerate(batch):
        n = doc.num_sentences
        input_ids[i, :n] = torch.stack(doc.input_ids)
        attn_mask[i, :n] = torch.stack(doc.attention_masks)
        sent_mask[i, :n] = True
        pos_feats[i, :n] = doc.positional_features
        labels[i, :n]    = doc.labels
    return {"input_ids": input_ids, "attention_mask": attn_mask,
            "sentence_mask": sent_mask, "pos_features": pos_feats, "labels": labels}


# ==========================================
# 4. Model Definition
# ==========================================
SENT_CHUNK = 32  # encode this many sentences through BERT at once to avoid OOM

@dataclass
class SegmenterConfig:
    encoder_model_name: str    = "nlpaueb/legal-bert-base-uncased"
    freeze_encoder_epochs: int = 2
    pos_feature_dim: int       = 4
    proj_hidden_size: int      = 256
    doc_encoder_type: str      = "transformer"
    doc_encoder_layers: int    = 2
    doc_encoder_heads: int     = 4
    doc_encoder_ff_dim: int    = 512
    doc_encoder_dropout: float = 0.1
    num_labels: int            = 2
    head_dropout: float        = 0.2
    use_crf: bool              = True
    use_focal_loss: bool       = False
    focal_gamma: float         = 2.0
    boundary_pos_weight: float = 8.0

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__(); self.gamma = gamma
    def forward(self, logits, targets):
        mask = targets != -100
        logits, targets = logits[mask], targets[mask]
        log_p = torch.log_softmax(logits, dim=-1)
        return nn.NLLLoss()((1 - torch.exp(log_p)) ** self.gamma * log_p, targets)

class TransformerDocEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(cfg.proj_hidden_size, cfg.doc_encoder_heads,
                cfg.doc_encoder_ff_dim, cfg.doc_encoder_dropout, batch_first=True),
            num_layers=cfg.doc_encoder_layers)
    def forward(self, x, mask):
        return self.encoder(x, src_key_padding_mask=mask)

class HierarchicalClauseSegmenter(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg
        self.sentence_encoder = AutoModel.from_pretrained(cfg.encoder_model_name)
        H = self.sentence_encoder.config.hidden_size
        self.feature_proj = nn.Sequential(
            nn.Linear(H + cfg.pos_feature_dim, cfg.proj_hidden_size),
            nn.LayerNorm(cfg.proj_hidden_size), nn.GELU())
        self.doc_encoder = TransformerDocEncoder(cfg)
        self.head = nn.Sequential(
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.proj_hidden_size, cfg.num_labels))
        self.crf = CRF(cfg.num_labels, batch_first=True) if cfg.use_crf else None
        if cfg.use_focal_loss:
            self.loss_fn = FocalLoss(cfg.focal_gamma)
        else:
            # register weight as a buffer so it moves to GPU with .to(device)
            self.register_buffer(
                "ce_weight", torch.tensor([1.0, cfg.boundary_pos_weight]))
            self.loss_fn = None  # built on-the-fly in forward using ce_weight

    def _encode(self, input_ids, attn_mask):
        # chunked: encode SENT_CHUNK sentences at a time, cat CLS embeddings
        B, N, L = input_ids.shape
        chunks = []
        for i in range(0, N, SENT_CHUNK):
            id_c   = input_ids[:, i:i+SENT_CHUNK, :].reshape(-1, L)
            mask_c = attn_mask[:, i:i+SENT_CHUNK, :].reshape(-1, L)
            out    = self.sentence_encoder(input_ids=id_c, attention_mask=mask_c)
            cls    = out.last_hidden_state[:, 0, :]
            chunks.append(cls.view(B, -1, cls.shape[-1]))
        return torch.cat(chunks, dim=1)

    def forward(self, input_ids, attention_mask, sentence_mask,
                pos_features, labels=None):
        cls    = self._encode(input_ids, attention_mask)
        proj   = self.feature_proj(torch.cat([cls, pos_features], -1))
        ctx    = self.doc_encoder(proj, ~sentence_mask)
        logits = self.head(ctx)
        loss   = None
        if labels is not None:
            if self.crf is not None:
                safe = labels.clone(); safe[safe == -100] = 0
                # CRF requires float32 — cast explicitly so AMP doesn't break it
                loss = -self.crf(logits.float(), safe,
                                 mask=sentence_mask.bool(), reduction="mean")
            elif self.loss_fn is not None:
                # FocalLoss path
                B, N, C = logits.shape
                loss = self.loss_fn(logits.view(B*N, C), labels.view(B*N))
            else:
                # CrossEntropy path — use registered buffer weight (lives on GPU)
                B, N, C = logits.shape
                loss = nn.functional.cross_entropy(
                    logits.float().view(B*N, C), labels.view(B*N),
                    weight=self.ce_weight, ignore_index=-100)
        return loss, logits

    @torch.no_grad()
    def predict(self, input_ids, attention_mask, sentence_mask,
                pos_features, threshold=0.5):
        _, logits = self.forward(input_ids, attention_mask,
                                 sentence_mask, pos_features)
        if self.crf is not None:
            preds = self.crf.decode(logits, mask=sentence_mask.bool())
            B, N  = logits.shape[:2]
            out   = torch.zeros(B, N, dtype=torch.long)
            for i, seq in enumerate(preds):
                out[i, :len(seq)] = torch.tensor(seq)
            return out
        return (torch.softmax(logits, -1)[:, :, 1] >= threshold).long()

    def freeze_encoder(self):
        for p in self.sentence_encoder.parameters(): p.requires_grad = False
    def unfreeze_encoder(self):
        for p in self.sentence_encoder.parameters(): p.requires_grad = True


# ==========================================
# 5. Metrics & Evaluation
# ==========================================
def _prf(preds, labels):
    tp = fp = fn = 0
    for p, g in zip(preds, labels):
        if   p==1 and g==1: tp+=1
        elif p==1 and g==0: fp+=1
        elif p==0 and g==1: fn+=1
    pr = tp/(tp+fp+1e-8); rc = tp/(tp+fn+1e-8)
    return {"precision": pr, "recall": rc,
            "boundary_f1": 2*pr*rc/(pr+rc+1e-8)}

def _window_diff(ref, hyp, k=None):
    N = len(ref)
    if N < 2: return 0.0
    k = k or max(1, N // (2*max(sum(ref), 1)))
    return sum(sum(ref[i+1:i+k+1]) != sum(hyp[i+1:i+k+1])
               for i in range(N-k)) / max(N-k, 1)

def compute_metrics(all_preds, all_labels):
    fp, fl, wd = [], [], []
    for ps, ls in zip(all_preds, all_labels):
        valid = [(p,l) for p,l in zip(ps,ls) if l != -100]
        if not valid: continue
        p_seq, l_seq = zip(*valid)
        fp.extend(p_seq); fl.extend(l_seq)
        if len(l_seq) > 1:
            wd.append(_window_diff(list(l_seq), list(p_seq)))
    m = _prf(fp, fl)
    m["window_diff"] = sum(wd)/max(len(wd), 1)
    return m

def find_best_threshold(all_probs, all_labels, steps=20):
    best_t, best_val, best_m = 0.5, -1.0, {}
    for i in range(steps+1):
        t = i/steps
        m = compute_metrics(
            [[int(p>=t) for p in ps] for ps in all_probs], all_labels)
        if m["boundary_f1"] > best_val:
            best_val, best_t, best_m = m["boundary_f1"], t, m
    return best_t, best_m


# ==========================================
# 6. CUAD Loader & Data Prep
# ==========================================
@dataclass
class CUADContract:
    doc_id: str
    text: str
    clauses: List[Tuple[int, int]]
    clause_types: Dict[int, List[str]] = field(default_factory=dict)

def _merge_spans(spans):
    if not spans: return []
    spans = sorted(spans); m = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= m[-1][1] + 1: m[-1][1] = max(m[-1][1], e)
        else: m.append([s, e])
    return [tuple(x) for x in m]

def load_cuad(cuad_root, max_contracts=None):
    cuad_root = Path(cuad_root)
    json_path = cuad_root / "CUAD_v1.json"
    if not json_path.exists():
        raise FileNotFoundError(f"CUAD_v1.json not found at {json_path}")

    print(f"Reading {json_path}  ({json_path.stat().st_size // 1_000_000} MB) ...")
    data = json.loads(json_path.read_bytes())["data"]
    print(f"  {len(data)} contracts in JSON")
    if max_contracts:
        data = data[:max_contracts]

    contracts = []
    for entry in tqdm(data, desc="Building contracts", unit="doc", dynamic_ncols=True):
        doc_id    = entry["title"].replace(".pdf","").replace(".txt","").strip()
        para      = entry["paragraphs"][0]
        full_text = para["context"]

        all_spans = []
        for qa in para["qas"]:
            for ans in qa.get("answers", []):
                a_text, a_start = ans["text"], ans["answer_start"]
                if not a_text.strip(): continue
                a_end = a_start + len(a_text)
                if full_text[a_start:a_end] == a_text:
                    all_spans.append((a_start, a_end))
                else:
                    lo  = max(0, a_start - 10)
                    idx = full_text.find(a_text, lo, a_end + 10)
                    if idx != -1:
                        all_spans.append((idx, idx + len(a_text)))

        merged = _merge_spans(all_spans)
        if not merged: continue
        contracts.append(CUADContract(doc_id=doc_id, text=full_text, clauses=merged))

    print(f"Loaded {len(contracts)}/{len(data)} contracts with annotations")
    return contracts

def _contract_type(doc_id):
    parts = doc_id.replace("-","_").split("_")
    return parts[-1] if parts else "Unknown"

def stratified_split(contracts, val_frac=0.10, test_frac=0.10, seed=42):
    rng = random.Random(seed)
    by_type = defaultdict(list)
    for c in contracts: by_type[_contract_type(c.doc_id)].append(c)
    train, val, test = [], [], []
    for group in by_type.values():
        rng.shuffle(group); n = len(group)
        nv = min(max(1, round(n*val_frac)),  n//3) if n>=3 else 0
        nt = min(max(1, round(n*test_frac)), n//3) if n>=3 else 0
        test  += group[:nt]
        val   += group[nt:nt+nv]
        train += group[nt+nv:]
    rng.shuffle(train)
    return train, val, test

def write_jsonl(contracts, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    skipped = 0
    with open(path, "w", encoding="utf-8") as f:
        for c in tqdm(contracts, desc=f"Writing {Path(path).name}", leave=False):
            try:
                f.write(json.dumps({
                    "doc_id":  c.doc_id,
                    "text":    _sanitize(c.text),
                    "clauses": [list(s) for s in c.clauses],
                }, ensure_ascii=True) + "\n")
            except Exception as e:
                skipped += 1
                print(f"  Skipping {c.doc_id}: {e}")
    print(f"  Wrote {len(contracts)-skipped} contracts → {path}"
          + (f"  ({skipped} skipped)" if skipped else ""))


# ==========================================
# 7. Training Loop  (AMP + DataParallel)
# ==========================================
def build_optimiser(model, encoder_lr=2e-5, head_lr=1e-3, weight_decay=0.01):
    enc_ids = {id(p) for p in model.sentence_encoder.parameters()}
    return AdamW([
        {"params": model.sentence_encoder.parameters(), "lr": encoder_lr},
        {"params": [p for p in model.parameters() if id(p) not in enc_ids],
         "lr": head_lr},
    ], weight_decay=weight_decay, betas=(0.9, 0.999))

def train_epoch(model, loader, optimiser, device, grad_accum=8, scaler=None):
    model.train(); total_loss = 0.0; optimiser.zero_grad()
    pbar = tqdm(loader, desc="  Train", leave=False, unit="batch")

    for step, batch in enumerate(pbar):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            loss, _ = model(batch["input_ids"], batch["attention_mask"],
                            batch["sentence_mask"], batch["pos_features"],
                            batch["labels"])

        # DataParallel returns per-GPU losses — reduce to scalar
        if loss is not None and loss.dim() > 0:
            loss = loss.mean()

        loss = loss / grad_accum

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_loss += loss.item() * grad_accum

        if (step+1) % grad_accum == 0:
            if scaler is not None:
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimiser)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
            optimiser.zero_grad()

        pbar.set_postfix({"loss": f"{loss.item() * grad_accum:.4f}"})

    # flush remaining gradients at end of epoch
    if len(loader) % grad_accum != 0:
        if scaler is not None:
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimiser)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
        optimiser.zero_grad()

    return total_loss / len(loader)

@torch.no_grad()
def eval_epoch(model, loader, device, threshold=0.5):
    model.eval(); all_preds, all_labels = [], []
    # unwrap DataParallel to call .predict()
    base_model = model.module if hasattr(model, "module") else model

    for batch in tqdm(loader, desc="  Eval ", leave=False, unit="batch"):
        batch = {k: v.to(device) for k, v in batch.items()}
        preds = base_model.predict(
            batch["input_ids"], batch["attention_mask"],
            batch["sentence_mask"], batch["pos_features"], threshold)
        for i in range(preds.shape[0]):
            n = batch["sentence_mask"][i].sum().item()
            all_preds.append(preds[i, :n].cpu().tolist())
            all_labels.append(batch["labels"][i, :n].cpu().tolist())
    return compute_metrics(all_preds, all_labels)

def collect_probs(model, loader, device):
    """Collect per-sentence boundary probabilities for threshold sweep."""
    model.eval()
    base_model = model.module if hasattr(model, "module") else model
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Probs", leave=False):
            batch   = {k: v.to(device) for k, v in batch.items()}
            _, lgts = base_model(batch["input_ids"], batch["attention_mask"],
                                 batch["sentence_mask"], batch["pos_features"])
            probs = torch.softmax(lgts, -1)[:, :, 1]
            for i in range(probs.shape[0]):
                n = batch["sentence_mask"][i].sum().item()
                all_probs.append(probs[i, :n].cpu().tolist())
                all_labels.append(batch["labels"][i, :n].cpu().tolist())
    return all_probs, all_labels

def save_ckpt(model, tokenizer, cfg, path):
    path = Path(path); path.mkdir(parents=True, exist_ok=True)
    # always save the unwrapped base model
    base = model.module if hasattr(model, "module") else model
    torch.save(base.state_dict(), path/"model.pt")
    tokenizer.save_pretrained(str(path))
    with open(path/"config.json", "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)


# ==========================================
# 8. Main Execution
# ==========================================
# ── Config  (tuned for 2×T4 Kaggle) ──────────────────────────────────────────
CUAD_DIR      = "/kaggle/input/datasets/arnav4124/cuad-dataset/CUAD_v1/CUAD_v1"
OUTPUT_DIR    = "/kaggle/working/exp1"
ENCODER       = "nlpaueb/legal-bert-base-uncased"
EPOCHS        = 10          # more room to improve from 0.597 baseline
BATCH_SIZE    = 2           # per GPU; effective = BATCH_SIZE * num_gpus * GRAD_ACCUM
GRAD_ACCUM    = 8
ENCODER_LR    = 2e-5
HEAD_LR       = 1e-3
FREEZE_EPOCHS = 1           # unfreeze sooner for more fine-tuning
PATIENCE      = 4
MAX_SENT_LEN  = 64
MAX_SENTS     = 200
MAX_CONTRACTS = None        # set e.g. 20 for a quick smoke test
# ─────────────────────────────────────────────────────────────────────────────

data_dir   = Path(OUTPUT_DIR) / "data"
train_path = data_dir / "train.jsonl"
val_path   = data_dir / "val.jsonl"
test_path  = data_dir / "test.jsonl"
ckpt_path  = Path(OUTPUT_DIR) / "best_model"

gc.collect(); torch.cuda.empty_cache()
if torch.cuda.is_available():
    print(f"GPU free at start: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")

# ── Step 1: Data Preparation ──────────────────────────────────────────────────
if not train_path.exists():
    print("\n=== Step 1: Data Preparation ===")
    t0 = time.time()
    contracts = load_cuad(CUAD_DIR, max_contracts=MAX_CONTRACTS)
    train_c, val_c, test_c = stratified_split(contracts)
    print(f"Split → train={len(train_c)}, val={len(val_c)}, test={len(test_c)}")
    write_jsonl(train_c, train_path)
    write_jsonl(val_c,   val_path)
    write_jsonl(test_c,  test_path)
    print(f"Data prep done in {(time.time()-t0)/60:.1f} min")
else:
    print("JSONL files exist — skipping data prep.")

# ── Step 2: Tokenise ──────────────────────────────────────────────────────────
print("\n=== Step 2: Tokenising ===")
tokenizer = AutoTokenizer.from_pretrained(ENCODER)
train_ds  = ContractDataset.from_jsonl(train_path, tokenizer, MAX_SENT_LEN, MAX_SENTS)
val_ds    = ContractDataset.from_jsonl(val_path,   tokenizer, MAX_SENT_LEN, MAX_SENTS)
print(f"Train: {len(train_ds)} docs  |  Val: {len(val_ds)} docs")

train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                          collate_fn=collate_contracts, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False,
                          collate_fn=collate_contracts, num_workers=2)

# ── Step 3: Build Model ───────────────────────────────────────────────────────
print("\n=== Step 3: Building model ===")
cfg   = SegmenterConfig(encoder_model_name=ENCODER,
                        freeze_encoder_epochs=FREEZE_EPOCHS, use_crf=True)
model = HierarchicalClauseSegmenter(cfg)

# gradient checkpointing: trades ~30% speed for ~40% less VRAM
model.sentence_encoder.gradient_checkpointing_enable()
print("Gradient checkpointing enabled")

# multi-GPU via DataParallel
if torch.cuda.device_count() > 1:
    print(f"DataParallel across {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

model = model.to(DEVICE)
base_model = model.module if hasattr(model, "module") else model

total_params = sum(p.numel() for p in base_model.parameters()) / 1e6
print(f"Parameters: {total_params:.1f} M")

if FREEZE_EPOCHS > 0:
    base_model.freeze_encoder()
    print(f"Encoder frozen for first {FREEZE_EPOCHS} epoch(s)")

optimiser = build_optimiser(base_model, ENCODER_LR, HEAD_LR)
scheduler = CosineAnnealingLR(optimiser, T_max=EPOCHS, eta_min=1e-6)
scaler    = torch.amp.GradScaler("cuda")

# ── Step 4: Train ─────────────────────────────────────────────────────────────
print("\n=== Step 4: Training ===")
best_f1, patience_cnt = 0.0, 0
epoch_bar = tqdm(range(1, EPOCHS+1), desc="Epochs", unit="epoch")

for epoch in epoch_bar:
    if epoch == FREEZE_EPOCHS + 1:
        base_model.unfreeze_encoder()
        print("Encoder unfrozen")

    gc.collect(); torch.cuda.empty_cache()

    t0          = time.time()
    train_loss  = train_epoch(model, train_loader, optimiser, DEVICE, GRAD_ACCUM, scaler)
    val_metrics = eval_epoch(model, val_loader, DEVICE, threshold=0.5)
    scheduler.step()

    f1  = val_metrics["boundary_f1"]
    eta = (time.time() - t0) / 60
    epoch_bar.set_postfix({"loss": f"{train_loss:.4f}", "F1": f"{f1:.4f}"})
    print(f"Epoch {epoch:02d} | loss={train_loss:.4f} | "
          f"P={val_metrics['precision']:.4f}  "
          f"R={val_metrics['recall']:.4f}  "
          f"F1={f1:.4f} | "
          f"WD={val_metrics['window_diff']:.4f} | {eta:.1f} min")

    if f1 > best_f1:
        best_f1, patience_cnt = f1, 0
        save_ckpt(model, tokenizer, cfg, ckpt_path)
        print(f"  ✓ Checkpoint saved (F1={best_f1:.4f})")
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print("Early stopping."); break

print(f"\nTraining done. Best val F1: {best_f1:.4f}")

# ── Step 5: Threshold Sweep ───────────────────────────────────────────────────
print("\n=== Step 5: Threshold Sweep ===")
all_probs, all_val_labels = collect_probs(model, val_loader, DEVICE)
BEST_THRESHOLD, best_tm   = find_best_threshold(all_probs, all_val_labels)
print(f"Best threshold : {BEST_THRESHOLD:.2f}")
print(f"Val  F1        : {best_tm['boundary_f1']:.4f}")
print(f"Val  WD        : {best_tm['window_diff']:.4f}")

# ── Step 6: Test Evaluation ───────────────────────────────────────────────────
print("\n=== Step 6: Test Evaluation ===")
test_ds     = ContractDataset.from_jsonl(test_path, tokenizer, MAX_SENT_LEN, MAX_SENTS)
test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False,
                         collate_fn=collate_contracts, num_workers=2)
test_metrics = eval_epoch(model, test_loader, DEVICE, threshold=BEST_THRESHOLD)

print("\n── Final Test Results ──────────────────────")
for k, v in test_metrics.items():
    print(f"  {k:20s}: {v:.4f}")

# ── Save Results ──────────────────────────────────────────────────────────────
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
with open(Path(OUTPUT_DIR)/"test_results.json", "w") as f:
    json.dump(test_metrics, f, indent=2)
with open(Path(OUTPUT_DIR)/"best_threshold.json", "w") as f:
    json.dump({"threshold": BEST_THRESHOLD, "metrics": best_tm}, f, indent=2)

print(f"\nAll done! Results saved to {OUTPUT_DIR}")
print(f"Best threshold : {BEST_THRESHOLD:.2f}")
print(f"Test F1        : {test_metrics['boundary_f1']:.4f}")
print(f"Test WD        : {test_metrics['window_diff']:.4f}")


# ==========================================
# 9. Inference Wrapper (ClauseSegmenter)
# ==========================================
@dataclass
class ClauseSpan:
    index: int
    text: str
    start_char: int
    end_char: int

class ClauseSegmenter:
    """
    Thin inference wrapper around HierarchicalClauseSegmenter.
    Accepts raw contract text, returns list of ClauseSpan objects.
    Used by Stage 2 ConstraintPipeline as stage1_segmenter.
    """
    def __init__(self, model, tokenizer, device,
                 threshold=0.5, max_sent_len=64):
        self.model      = model
        self.tokenizer  = tokenizer
        self.device     = device
        self.threshold  = threshold
        self.max_sent_len = max_sent_len
        self.model.eval().to(device)

    def segment(self, text: str) -> List[ClauseSpan]:
        sents = split_contract(text)
        if not sents: return []

        # tokenise
        ids_list, mask_list = [], []
        for s in sents:
            enc = self.tokenizer(
                s.text, max_length=self.max_sent_len,
                padding="max_length", truncation=True, return_tensors="pt")
            ids_list.append(enc["input_ids"].squeeze(0))
            mask_list.append(enc["attention_mask"].squeeze(0))

        N       = len(sents)
        seq_len = ids_list[0].shape[0]
        input_ids = torch.stack(ids_list).unsqueeze(0).to(self.device)   # [1,N,L]
        attn_mask = torch.stack(mask_list).unsqueeze(0).to(self.device)  # [1,N,L]
        sent_mask = torch.ones(1, N, dtype=torch.bool).to(self.device)   # [1,N]
        pos_feats = _pos_features(sents).unsqueeze(0).to(self.device)    # [1,N,4]

        preds  = self.model.predict(input_ids, attn_mask,
                                    sent_mask, pos_feats, self.threshold)
        labels = preds[0, :N].cpu().tolist()
        return self._build_spans(text, sents, labels)

    @staticmethod
    def _build_spans(text, sents, labels) -> List[ClauseSpan]:
        spans, ci, cs, cur = [], 0, None, []
        def flush(end):
            nonlocal ci, cs, cur
            if cs is None: return
            spans.append(ClauseSpan(ci, " ".join(cur), cs, end))
            ci += 1; cs = None; cur = []
        for sent, label in zip(sents, labels):
            if label == 1:
                flush(sent.start_char); cs = sent.start_char; cur = [sent.text]
            else:
                if cs is None: cs = sent.start_char
                cur.append(sent.text)
        flush(len(text))
        return spans

    @classmethod
    def from_pretrained(cls, ckpt_dir, device=None, threshold=0.5,
                        max_sent_len=64):
        ckpt_dir = Path(ckpt_dir)
        device   = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        cfg      = SegmenterConfig(**json.load(open(ckpt_dir / "config.json")))
        m        = HierarchicalClauseSegmenter(cfg)
        m.load_state_dict(
            torch.load(ckpt_dir / "model.pt", map_location=device))
        tok = AutoTokenizer.from_pretrained(str(ckpt_dir))
        return cls(m, tok, device, threshold, max_sent_len)


# ==========================================
# 10. Generate Stage 2 Predictions
# ==========================================
# Run the trained model on raw contract text from every split.
# Output files use PREDICTED clause boundaries (not gold spans),
# so Stage 2 trains on realistic noisy input — the same quality
# it will see at inference time.
#
# Output folder layout expected by Stage 2:
#   /kaggle/working/stage1_predictions/
#       predicted_train.jsonl
#       predicted_val.jsonl
#       predicted_test.jsonl
#
# Upload that folder to Kaggle as a dataset named "stage1-predictions".
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== Step 7: Generating Stage 2 Predictions ===")

# Load the best checkpoint we just saved
segmenter = ClauseSegmenter.from_pretrained(
    ckpt_path,
    device=DEVICE,
    threshold=BEST_THRESHOLD,
    max_sent_len=MAX_SENT_LEN,
)
segmenter.model.eval()

pred_dir = Path("/kaggle/working/stage1_predictions")
pred_dir.mkdir(parents=True, exist_ok=True)

def generate_predictions(jsonl_path: Path, out_path: Path,
                          seg: ClauseSegmenter) -> None:
    """
    Read each contract from jsonl_path (using only doc_id + text,
    ignoring gold clauses), run Stage 1 segmenter, write predicted
    clause boundaries to out_path.
    """
    lines = jsonl_path.read_bytes().decode("utf-8", errors="replace").splitlines()
    written = skipped = 0

    with open(out_path, "w", encoding="utf-8") as f_out:
        for line in tqdm(lines,
                         desc=f"Predicting → {out_path.name}",
                         unit="doc", dynamic_ncols=True):
            if not line.strip(): continue
            try:
                obj      = json.loads(line)
                doc_id   = obj["doc_id"]
                text     = obj["text"]

                # ── Stage 1 inference on raw text ─────────────────────────────
                # Gold clauses (obj["clauses"]) are deliberately ignored here.
                # We want what the model predicts, not ground truth.
                pred_spans = seg.segment(text)

                f_out.write(json.dumps({
                    "doc_id":  doc_id,
                    "text":    text,
                    # list of [start_char, end_char] — predicted boundaries
                    "clauses": [[c.start_char, c.end_char]
                                for c in pred_spans],
                    # optional metadata useful for Stage 2 debugging
                    "num_predicted_clauses": len(pred_spans),
                }, ensure_ascii=True) + "\n")
                written += 1

            except Exception as e:
                skipped += 1
                print(f"  Skipping line: {e}")

    print(f"  {out_path.name}: {written} docs written"
          + (f", {skipped} skipped" if skipped else ""))

# Generate for all three splits
generate_predictions(train_path, pred_dir / "predicted_train.jsonl", segmenter)
generate_predictions(val_path,   pred_dir / "predicted_val.jsonl",   segmenter)
generate_predictions(test_path,  pred_dir / "predicted_test.jsonl",  segmenter)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\nStage 2 prediction files saved to: {pred_dir}")
for f in sorted(pred_dir.iterdir()):
    lines = sum(1 for _ in open(f))
    size  = f.stat().st_size // 1_000_000
    print(f"  {f.name:<35s}  {lines} docs  ({size} MB)")

print("""
Next steps:
  1. Go to Kaggle → Datasets → New Dataset
  2. Upload the folder:  /kaggle/working/stage1_predictions/
  3. Name the dataset:   stage1-predictions
  4. In Stage 2 notebook Cell 3, set:
       TRAIN_JSONL = "/kaggle/input/stage1-predictions/predicted_train.jsonl"
       VAL_JSONL   = "/kaggle/input/stage1-predictions/predicted_val.jsonl"
       TEST_JSONL  = "/kaggle/input/stage1-predictions/predicted_test.jsonl"

""")
