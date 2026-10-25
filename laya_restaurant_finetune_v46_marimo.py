# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "torch",
#     "numpy",
#     "matplotlib",
#     "packaging",
#     "transformers>=4.41,<5.0",
#     "datasets>=2.19,<3.0",
#     "accelerate>=1.1,<2.0",
#     "peft>=0.11,<0.12",
#     "evaluate>=0.4",
#     "scikit-learn>=1.3",
#     "sentencepiece>=0.2",
#     "huggingface_hub>=0.34,<1.0",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # فاین‌تیون مدل فارسی برای دستیار تلفنی رستوران (Intent + NER + Clarification)

    **نسخه v46** — دیتاست تقویت‌شده و مقاوم‌سازی نهایی، ۵۶ قصد و نیت، مقاوم‌سازی کامل در برابر خطاهای STT و لهجه‌های فارسی، نظارت کامل بر زیرواژه‌ها در NER، پالایش خودکار اموجی در استنتاج، و تست مهروموم‌شده.

    این نوت‌بوک با `sarv_kalchi_nlu_dataset_v46.jsonl` و splitهای صریح
    `v46/splits_v46/{train,val,test}.jsonl` کار می‌کند؛ test کاملاً مهروموم است
    و هش SHA-256 آن با `SPLIT_MANIFEST.json` اعتبارسنجی می‌شود.

    **هدف:** تحلیل پیام صوتی/متنی مشتری و تولید خروجی ساختاریافته
    `(intent, secondary_intent, entities, action, needs_clarification)`.

    **معماری:** backbone مشترک ParsBERT + LoRA و چهار سر پیش‌بینی:
    1. Intent classification (۵۶ کلاس در نسخه v46)
    2. Secondary-intent classification (۵۶ کلاس + none)
    3. Needs-clarification (باینری برای تشخیص ابهام و عدم قطعیت)
    4. NER به روش Token Classification / BIO (۱۲ موجودیت شامل FOOD, DRINK, DESSERT, QUANTITY, TIME, DATE, ADDRESS, PAYMENT_METHOD, ...)

    ---
    **نسخه‌ی marimo/molab:** این نسخه از نوتبوک Colab تبدیل شده است. روی molab باید GPU را از دکمه‌ی مشخصات نوتبوک (بالای صفحه) روشن کنید. ZIP دیتاست (`sarv_kalchi_nlu_dataset_v46_enriched.zip`) را یا کنار نوتبوک بگذارید یا در سلول تنظیمات `DATASET_ZIP_URL` را پر کنید. سلول‌های اختیاری Auto-Fix و K-Fold (در Colab سلول ۲۷ و ۲۸؛ غیرفعال با `ALLOW_AUTO_FIX=False`) در فایل جدا `optional_autofix_kfold_cells.py` هستند.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # نسخه v46

    - ۳۷٬۴۲۶ رکورد جدید برای اینتنت‌های ضعیف، حذف تعارض برچسب، place_order از ۵۱k به ۲۵k کاهش یافت
    """)
    return


@app.cell
def _():
    # CELL 3/4: بررسی نسخه‌ها + import کتابخانه‌ها
    # نکته‌ی marimo: هر نام سراسری فقط در یک سلول تعریف می‌شود؛ پس همه‌ی importها اینجا جمع شده‌اند.
    import os
    import sys
    import json
    import glob
    import copy
    import time
    import zlib
    import gc
    import hashlib
    import platform
    import random
    import re
    import shutil
    import zipfile
    import urllib.request
    from collections import Counter, defaultdict

    import numpy as np
    import torch
    import torch.nn as nn
    import transformers
    import datasets
    import accelerate
    import peft
    from packaging.version import parse as parse_version
    from torch.utils.data import DataLoader
    from transformers import (
        AutoTokenizer, AutoModel, AutoConfig,
        TrainingArguments, Trainer, PreTrainedModel,
        DataCollatorWithPadding,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        classification_report, confusion_matrix,
    )

    print("transformers:", transformers.__version__)
    print("datasets    :", datasets.__version__)
    print("torch       :", torch.__version__)
    print("peft        :", peft.__version__)
    print("accelerate  :", accelerate.__version__)

    # این کد با transformers 4.x و peft 0.11.x نوشته شده (پین‌ها در هدر PEP 723 فایل هست).
    if parse_version(transformers.__version__) >= parse_version("5.0.0"):
        print("⚠️ transformers 5.x نصب است؛ نوتبوک برای 4.41–4.x نوشته شده. اگر خطا دیدید، نسخه را پین کنید.")
    return (
        AutoConfig,
        AutoModel,
        AutoTokenizer,
        Counter,
        DataLoader,
        Dataset,
        LoraConfig,
        TaskType,
        Trainer,
        TrainingArguments,
        accuracy_score,
        classification_report,
        confusion_matrix,
        defaultdict,
        get_peft_model,
        glob,
        hashlib,
        json,
        nn,
        np,
        os,
        parse_version,
        platform,
        precision_recall_fscore_support,
        random,
        re,
        shutil,
        sys,
        torch,
        transformers,
        urllib,
        zipfile,
    )


@app.cell
def _(platform, sys, torch):
    # CELL 2: بررسی GPU / CUDA
    print("Python version:", sys.version)
    print("Platform:", platform.platform())
    print("\nPyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("GPU name:", torch.cuda.get_device_name(0))
        _props = torch.cuda.get_device_properties(0)
        print(f"VRAM total: {_props.total_memory / 1024**3:.1f} GB")
    else:
        print("\n⚠️ GPU در دسترس نیست! آموزش روی CPU بسیار کند است.")
        print("   در molab: دکمه‌ی مشخصات نوتبوک در هدر → روشن کردن GPU")
    return


@app.cell
def _(np, os, random, torch):
    # CELL 5: تنظیمات (Config) — دیتاست v46
    SEED = 42

    # --- backbone ---
    BASE_MODEL_NAME = "HooshvareLab/bert-fa-base-uncased"
    MAX_SEQ_LEN = 64
    MAX_SEQ_LEN_WITH_CONTEXT = 96

    # --- paths ---
    DATA_DIR = "./v46"
    SPLITS_DIR = os.path.join(DATA_DIR, "splits_v46")
    OUTPUT_DIR = "./parsbert-restaurant-fa-v46-working"
    FINAL_MODEL_DIR = "./parsbert-restaurant-fa-v46-final"

    DATASET_ZIP_URL = ""   # اگر ZIP دیتاست را جایی آپلود کرده‌اید (لینک مستقیم)، اینجا بگذارید؛ وگرنه ZIP را کنار نوتبوک بگذارید
    MANIFEST_FILENAME = "SPLIT_MANIFEST.json"
    STT_EVAL_FILENAME = "sarv_kalchi_nlu_stt_eval_v46.jsonl"

    # --- LoRA ---
    LORA_R = 32
    LORA_ALPHA = 64
    LORA_DROPOUT = 0.1
    LORA_TARGET_MODULES = ["query", "key", "value"]

    # --- training ---
    NUM_EPOCHS = 5
    LEARNING_RATE = 2e-4
    TRAIN_BATCH_SIZE = 16
    EVAL_BATCH_SIZE = 32
    GRADIENT_ACCUMULATION_STEPS = 8
    WARMUP_RATIO = 0.03
    WEIGHT_DECAY = 0.01
    SAVE_STEPS = 300
    EVAL_STEPS = 300
    LOGGING_STEPS = 100

    # --- multi-task loss weights ---
    W_INTENT = 1.0
    W_SECONDARY = 0.5
    W_CLARIFICATION = 0.5
    W_NER = 1.0

    # --- safety switch ---
    ALLOW_AUTO_FIX = False

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Config loaded for v46. SEED =", SEED)
    return (
        BASE_MODEL_NAME,
        DATASET_ZIP_URL,
        DATA_DIR,
        EVAL_BATCH_SIZE,
        EVAL_STEPS,
        FINAL_MODEL_DIR,
        GRADIENT_ACCUMULATION_STEPS,
        LEARNING_RATE,
        LOGGING_STEPS,
        LORA_ALPHA,
        LORA_DROPOUT,
        LORA_R,
        LORA_TARGET_MODULES,
        MANIFEST_FILENAME,
        MAX_SEQ_LEN,
        MAX_SEQ_LEN_WITH_CONTEXT,
        NUM_EPOCHS,
        OUTPUT_DIR,
        SAVE_STEPS,
        SEED,
        SPLITS_DIR,
        STT_EVAL_FILENAME,
        TRAIN_BATCH_SIZE,
        WARMUP_RATIO,
        WEIGHT_DECAY,
        W_CLARIFICATION,
        W_INTENT,
        W_NER,
        W_SECONDARY,
    )


@app.cell
def _(AutoConfig, AutoModel, AutoTokenizer, BASE_MODEL_NAME, torch):
    # CELL 7: بارگذاری backbone و tokenizer
    print(f"در حال بارگذاری backbone: {BASE_MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    backbone_config = AutoConfig.from_pretrained(BASE_MODEL_NAME)
    backbone = AutoModel.from_pretrained(BASE_MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone.to(device)

    print("Tokenizer vocab size:", tokenizer.vocab_size)
    print("Hidden size         :", backbone_config.hidden_size)
    print("Num layers           :", backbone_config.num_hidden_layers)
    print("Num attention heads  :", backbone_config.num_attention_heads)
    print("Device               :", device)
    return backbone, backbone_config, device, tokenizer


@app.cell
def _(backbone, nn):
    # CELL 8: بررسی معماری backbone (برای انتخاب صحیح LORA_TARGET_MODULES)
    _bb_total_params = sum(p.numel() for p in backbone.parameters())
    print(f"Total backbone parameters: {_bb_total_params:,}")

    print("\n--- نام ماژول‌های Linear در اولین لایه ترنسفورمر (برای تنظیم LoRA target_modules) ---")
    _first_layer_name = None
    for _name, _module in backbone.named_modules():
        if isinstance(_module, nn.Linear):
            print(_name)
            if _first_layer_name is None:
                __first_layer_name = _name

    print(
        "\nنکته: LORA_TARGET_MODULES در CELL 5 روی ['query', 'value'] تنظیم شده که برای معماری BERT/ParsBERT "
        "صحیح است. اگر backbone را عوض کردید، این لیست را بر اساس نام‌های چاپ‌شده در بالا به‌روز کنید."
    )
    return


@app.cell
def _(
    DATASET_ZIP_URL,
    DATA_DIR,
    MANIFEST_FILENAME,
    SPLITS_DIR,
    STT_EVAL_FILENAME,
    glob,
    hashlib,
    json,
    os,
    re,
    shutil,
    urllib,
    zipfile,
):
    # CELL 9 (v46): بارگذاری splitهای صریح، گروهی و مهروموم‌شده
    FA_CHAR_MAP = str.maketrans({"ك": "ک", "ي": "ی", "ى": "ی"})
    GROUP_FA_MAP = str.maketrans({"ك": "ک", "ي": "ی", "ى": "ی", "أ": "ا", "إ": "ا", "ة": "ه", "ۀ": "ه"})
    GROUP_CANON_MAP = str.maketrans({"ظ": "ز", "ض": "ز", "ذ": "ز", "ص": "س", "ث": "س",
                                     "ط": "ت", "ح": "ه", "ق": "غ"})


    def _norm_fa(s):
        return s.translate(FA_CHAR_MAP) if isinstance(s, str) else s


    def _standardize_bio(tags):
        """هر تگ B- تکراری را به I- تبدیل می‌کند (دیتاست v46 از قبل استاندارد است)."""
        out, previous_type = [], None
        for tag in tags:
            if tag == "O":
                out.append("O")
                previous_type = None
                continue
            if not isinstance(tag, str) or "-" not in tag:
                raise ValueError(f"NER tag نامعتبر: {tag!r}")
            prefix, entity_type = tag.split("-", 1)
            if prefix not in {"B", "I"}:
                raise ValueError(f"NER tag نامعتبر: {tag!r}")
            if prefix == "B" and previous_type == entity_type:
                prefix = "I"
            out.append(f"{prefix}-{entity_type}")
            previous_type = entity_type
        return out


    def normalize_record(r):
        history = r.get("history") or []
        context = [_norm_fa(h["text"]) for h in history if isinstance(h, dict) and h.get("text")]
        tokens = [_norm_fa(t) for t in (r.get("tokens") or [])]
        tags = _standardize_bio(r.get("ner_tags") or [])
        return {
            "id": r.get("id"),
            "text": _norm_fa(r.get("text", "")),
            "tokens": tokens,
            "ner_tags": tags,
            "intent": r.get("intent"),
            "secondary_intent": r.get("secondary_intent"),
            "needs_clarification": bool(r.get("needs_clarification", False)),
            "entities": r.get("entities") or {},
            "action": r.get("action"),
            "context": context,
            "conversation_id": r.get("conversation_id"),
            "history": history,
            "source": r.get("source"),
            "split": r.get("split"),
            "stt_error_types": r.get("stt_error_types"),
            "clean_text": r.get("clean_text"),
            "clean_intent": r.get("clean_intent"),
            "meaning_preserved": r.get("meaning_preserved"),
            "details_preserved": r.get("details_preserved"),
            "label_confidence": r.get("label_confidence"),
            "review_status": r.get("review_status"),
        }


    def _load_jsonl(path):
        rows = []
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{line_no}: JSON خراب — {e}") from e
        return rows


    def _data_ready():
        return (os.path.exists(os.path.join(SPLITS_DIR, "train.jsonl"))
                and os.path.exists(os.path.join(SPLITS_DIR, "val.jsonl"))
                and os.path.exists(os.path.join(SPLITS_DIR, "test.jsonl"))
                and os.path.exists(os.path.join(DATA_DIR, STT_EVAL_FILENAME))
                and os.path.exists(os.path.join(DATA_DIR, MANIFEST_FILENAME)))


    def _ensure_data_files():
        """اگر فایل‌ها محلی نیستند (مثلاً Colab)، همان ZIP نسخه v46 را می‌گیرد و باز می‌کند."""
        if _data_ready():
            return
        os.makedirs(DATA_DIR, exist_ok=True)
        candidates = (sorted(glob.glob(os.path.join(DATA_DIR, "*v46*.zip")))
                      + sorted(glob.glob("sarv_kalchi_nlu_dataset_v46*.zip")))
        if not candidates:
            if DATASET_ZIP_URL:
                print("دانلود ZIP دیتاست از DATASET_ZIP_URL ...")
                _dl = os.path.join(DATA_DIR, "sarv_kalchi_nlu_dataset_v46_enriched.zip")
                urllib.request.urlretrieve(DATASET_ZIP_URL, _dl)
                candidates = [_dl]
            else:
                try:
                    from google.colab import files as _colab_files
                except ImportError:
                    _colab_files = None
            if not candidates and _colab_files is None:
                raise FileNotFoundError(
                    f"فایل‌های v46 در «{DATA_DIR}» پیدا نشدند و ZIP هم در دسترس نیست.\n"
                    "فایل sarv_kalchi_nlu_dataset_v46_enriched.zip را کنار نوت‌بوک بگذارید، "
                    "یا DATASET_ZIP_URL را در سلول تنظیمات پر کنید."
                )
            if not candidates:
                print("ZIP دیتاست v46 را آپلود کنید: sarv_kalchi_nlu_dataset_v46_enriched.zip")
                uploaded = _colab_files.upload()
                candidates = [n for n in uploaded if n.lower().endswith(".zip")]
        if not candidates:
            raise RuntimeError("هیچ فایل ZIP آپلود نشد.")
        zip_path = candidates[0]
        print(f"در حال باز کردن {zip_path} در {DATA_DIR} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATA_DIR)
        # اگر ZIP یک پوشه‌ی والد داشته باشد، محتوا را یک سطح بالا می‌آوریم
        if not _data_ready():
            for d in [x for x in glob.glob(os.path.join(DATA_DIR, "*")) if os.path.isdir(x)]:
                if os.path.exists(os.path.join(d, "splits_v46")):
                    for item in os.listdir(d):
                        shutil.move(os.path.join(d, item), os.path.join(DATA_DIR, item))
                    break
        if not _data_ready():
            raise RuntimeError("محتوای ZIP با ساختار مورد انتظار v46 همخوان نیست.")
        print("✅ داده v46 از ZIP استخراج شد.")


    def _load_split(name):
        path = os.path.join(SPLITS_DIR, name + ".jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} پیدا نشد. فایل‌های data_v46/splits_v46 را آپلود کنید."
            )
        return [normalize_record(r) for r in _load_jsonl(path)]


    # --- اگر داده محلی نیست (Colab)، ZIP نسخه v46 را بگیر و باز کن ---
    _ensure_data_files()

    # --- تأیید مهروموم بودن test قبل از هر چیز ---
    _manifest_path = os.path.join(DATA_DIR, MANIFEST_FILENAME)
    _test_path = os.path.join(SPLITS_DIR, "test.jsonl")
    if os.path.exists(_manifest_path) and os.path.exists(_test_path):
        _manifest = json.load(open(_manifest_path, encoding="utf-8"))
        _hash = hashlib.sha256(open(_test_path, "rb").read()).hexdigest()
        assert _hash == _manifest["test_sha256"], (
            "فایل test.jsonl نسبت به manifest تغییر کرده است؛ ارزیابی نهایی معتبر نیست."
        )
        print("✅ هش test با SPLIT_MANIFEST مطابقت دارد:", _hash[:16], "…")
    else:
        print("⚠️ SPLIT_MANIFEST.json یا test.jsonl پیدا نشد؛ بررسی مهروموم انجام نشد.")

    raw_train = _load_split("train")
    raw_val = _load_split("val")
    raw_test = _load_split("test")
    stt_eval_raw = _load_jsonl(os.path.join(DATA_DIR, STT_EVAL_FILENAME))

    for _name, _rows in (("train", raw_train), ("val", raw_val), ("test", raw_test)):
        _bad = [r for r in _rows if not r["text"] or not r["intent"] or len(r["tokens"]) != len(r["ner_tags"])]
        if _bad:
            raise ValueError(f"{len(_bad)} رکورد نامعتبر در split {_name}")
        print(f"{_name}: {len(_rows):,}")

    print(f"STT eval: {len(stt_eval_raw):,}")


    def _group_text_key(value):
        value = str(value or "").translate(GROUP_FA_MAP).translate(GROUP_CANON_MAP).replace(chr(8204), " ")
        return " ".join(re.sub(r"[^\w]+", " ", value).split()).casefold()


    def build_group_ids(records):
        """گروه‌بندی برای بررسی نشت: گفتگو، یا متن+زمینه یکسان."""
        parent = list(range(len(records)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        first = {}
        for i, r in enumerate(records):
            keys = []
            if r.get("conversation_id"):
                keys.append(("conversation", str(r["conversation_id"])))
            text = _group_text_key(r.get("text", ""))
            ctx = tuple(" ".join(_norm_fa(x).split()) for x in (r.get("context") or []))
            keys.append(("turn", text, ctx))
            for key in keys:
                if key in first:
                    union(i, first[key])
                else:
                    first[key] = i
        roots, compact = [], {}
        for i in range(len(records)):
            root = find(i)
            if root not in compact:
                compact[root] = len(compact)
            roots.append(compact[root])
        return roots


    print("\n✅ splitهای مهروموم‌شده بارگذاری شدند.")
    return (
        FA_CHAR_MAP,
        build_group_ids,
        normalize_record,
        raw_test,
        raw_train,
        raw_val,
        stt_eval_raw,
    )


@app.cell
def _(Counter, build_group_ids, json, raw_test, raw_train, raw_val):
    # CELL 10: اعتبارسنجی دیتاست (validation)
    def validate(dataset, name):
        print(f"\n{'=' * 20} {name} {'=' * 20}")
        print(f'تعداد کل: {len(dataset)}')
        texts = [s['text'] for s in dataset]
        dup = [t for t, c in Counter(texts).items() if c > 1]
        print(f'Duplicate texts: {len(dup)}')
        no_text = sum((1 for s in dataset if not s.get('text')))
        no_intent = sum((1 for s in dataset if not s.get('intent')))
        print(f'Samples without text: {no_text}')
        print(f'Samples without intent: {no_intent}')
        bad_json = 0
        invalid_entities = 0
        len_mismatch = 0
        bad_bio_tag = 0
        for s in dataset:
            try:  # طول tokens با طول ner_tags یکی نیست
                json.dumps(s)  # تگی که نه "O" است و نه با B-/I- شروع می‌شود
            except Exception:
                bad_json = bad_json + 1
            if not isinstance(s.get('entities'), dict):
                invalid_entities = invalid_entities + 1
            if len(s.get('tokens', [])) != len(s.get('ner_tags', [])):
                len_mismatch = len_mismatch + 1
            for t in s.get('ner_tags', []):
                if t != 'O' and (not (t.startswith('B-') or t.startswith('I-'))):
                    bad_bio_tag = bad_bio_tag + 1
        print(f'Corrupted JSON records: {bad_json}')
        print(f'Invalid entities structure: {invalid_entities}')
        print(f'tokens/ner_tags با طول نامساوی: {len_mismatch}')
        print(f'BIO tag نامعتبر (نه O و نه B-/I-): {bad_bio_tag}')
        return {'total': len(dataset), 'duplicates': len(dup), 'no_text': no_text, 'no_intent': no_intent, 'bad_json': bad_json, 'invalid_entities': invalid_entities, 'len_mismatch': len_mismatch, 'bad_bio_tag': bad_bio_tag}
    print('توجه: مقادیر entities در این دیتاست نرمال\u200cشده\u200cاند (مثلا «دو تا» → QUANTITY:2، یا نام کامل یک غذا) و\nلزوما عینا در text ظاهر نمی\u200cشوند؛ پس دیگر چک «آیا entity عینا در متن هست؟» انجام نمی\u200cشود — NER به\u200cجای\nآن مستقیما از tokens/ner_tags خودِ دیتاست (که از قبل BIO-تگ شده) در CELL 14 استفاده می\u200cکند.')
    report_train = validate(raw_train, 'TRAIN')
    report_val = validate(raw_val, 'VAL')
    report_test = validate(raw_test, 'TEST')
    _all_split_rows = raw_train + raw_val + raw_test
    _all_group_ids = build_group_ids(_all_split_rows)
    _offsets = [0, len(raw_train), len(raw_train) + len(raw_val), len(_all_split_rows)]
    _groups_by_split = [set(_all_group_ids[_offsets[i]:_offsets[i + 1]]) for i in range(3)]
    assert not _groups_by_split[0] & _groups_by_split[1], 'نشت گروهی بین train و val پیدا شد.'
    assert not _groups_by_split[0] & _groups_by_split[2], 'نشت گروهی بین train و test پیدا شد.'
    assert not _groups_by_split[1] & _groups_by_split[2], 'نشت گروهی بین val و test پیدا شد.'
    print('\n✅ هیچ گفتگو یا متن+زمینه\u200cی یکسانی بین splitها مشترک نیست.')
    assert all((r['len_mismatch'] == 0 for r in [report_train, report_val, report_test])), 'برخی نمونه\u200cها طول tokens/ner_tags نامساوی دارند — این نباید اتفاق بیفتد چون CELL 9 این نمونه\u200cها را حذف می\u200cکند؛ اگر این خطا را دیدید، CELL 9 را دوباره اجرا کنید.'
    # Leakage check روی همه‌ی طول‌ها و هر سه split؛ conversation_id و متن+زمینه هر دو ملاک‌اند.
    # برای بررسی بین splitها، گروه‌ها را روی داده‌ی تجمیع‌شده می‌سازیم.
    print('\n✅ اعتبارسنجی کامل شد — مشکلی برای ادامه یافت نشد.' if all((r['duplicates'] == 0 and r['no_text'] == 0 and (r['bad_json'] == 0) and (r['bad_bio_tag'] == 0) for r in [report_train, report_val, report_test])) else '\n⚠️ برخی هشدارها بالا مشاهده شد — قبل از ادامه بررسی کنید.')
    return


@app.cell
def _(Counter, raw_test, raw_train, raw_val):
    # CELL 11: آمار دیتاست
    ALL_INTENTS = sorted(set((s['intent'] for s in raw_train + raw_val + raw_test)))
    print(f'تعداد intentهای منحصربه\u200cفرد: {len(ALL_INTENTS)}\n')
    intent_counts_train = Counter((s['intent'] for s in raw_train))
    print(f"{'intent':30s} {'train':>6} {'val':>6} {'test':>6}")
    intent_counts_val = Counter((s['intent'] for s in raw_val))
    intent_counts_test = Counter((s['intent'] for s in raw_test))
    for _intent in ALL_INTENTS:
        print(f'{_intent:30s} {intent_counts_train.get(_intent, 0):>6} {intent_counts_val.get(_intent, 0):>6} {intent_counts_test.get(_intent, 0):>6}')
    missing_in_val = [i for i in ALL_INTENTS if intent_counts_val.get(i, 0) == 0]
    missing_in_test = [i for i in ALL_INTENTS if intent_counts_test.get(i, 0) == 0]
    print(f"\nIntentهای غایب از val:  {missing_in_val or 'هیچ\u200cکدام'}")
    print(f"Intentهای غایب از test: {missing_in_test or 'هیچ\u200cکدام'}")
    n_clarif = sum((1 for s in raw_train if s.get('needs_clarification')))
    n_multi = sum((1 for s in raw_train if s.get('secondary_intent')))
    n_ctx = sum((1 for s in raw_train if s.get('context')))
    print(f'\n[train] needs_clarification=True : {n_clarif} ({n_clarif / len(raw_train):.1%})')
    print(f'[train] secondary_intent موجود    : {n_multi} ({n_multi / len(raw_train):.1%})')
    print(f'[train] context موجود             : {n_ctx} ({n_ctx / len(raw_train):.1%})')
    min_i, min_c = min(intent_counts_train.items(), key=lambda x: x[1])
    max_i, max_c = max(intent_counts_train.items(), key=lambda x: x[1])
    print(f'\nClass imbalance (train): کمترین = {min_i} ({min_c}), بیشترین = {max_i} ({max_c}), نسبت = {max_c / min_c:.1f}x')
    print('این imbalance طبیعی است (intentهای سفارش باید نمونه بیشتری داشته باشند)؛ در CELL 16 با class weighting جبران می\u200cشود.')
    return ALL_INTENTS, missing_in_test, missing_in_val


@app.cell
def _(missing_in_test, missing_in_val, raw_test, raw_train, raw_val):
    # CELL 12: تایید Train / Validation / Test split
    # split با StratifiedGroupKFold انجام شده و همه‌ی turnهای یک گفتگو و متن+زمینه‌ی تکراری کنار هم می‌مانند.
    # در صورت غیبت کلاس‌های نادر، فهرست missing را بررسی کنید و تعداد گروه‌های آن intent را افزایش دهید.
    _total = len(raw_train) + len(raw_val) + len(raw_test)
    print(f'Train: {len(raw_train)} ({len(raw_train) / _total:.1%})')
    print(f'Val:   {len(raw_val)} ({len(raw_val) / _total:.1%})')
    print(f'Test:  {len(raw_test)} ({len(raw_test) / _total:.1%})')
    print('هشدار: intentهای غایب از val:', missing_in_val or 'هیچ\u200cکدام')
    print('هشدار: intentهای غایب از test:', missing_in_test or 'هیچ\u200cکدام')
    print('\n✅ split گروهی تأیید شد؛ نبود کلاس نادر در گزارش بالا را بررسی کنید.')
    return


@app.cell
def _(ALL_INTENTS, raw_test, raw_train, raw_val):
    # CELL 13: ساخت label vocabulary (intent) و طرح BIO برای NER
    # --- intent labels ---
    intent2id = {intent: i for i, intent in enumerate(ALL_INTENTS)}
    id2intent = {i: intent for intent, i in intent2id.items()}
    NUM_INTENTS = len(ALL_INTENTS)
    SECONDARY_NONE = 'none'
    # --- secondary intent: همان فضای intent + یک کلاس "none" ---
    secondary2id = dict(intent2id)
    secondary2id[SECONDARY_NONE] = NUM_INTENTS
    id2secondary = {i: s for s, i in secondary2id.items()}
    NUM_SECONDARY = len(secondary2id)
    STRING_ENTITY_KEYS = sorted({tag.split('-', 1)[1] for s in raw_train + raw_val + raw_test for tag in s['ner_tags'] if tag != 'O'})
    print('Entity typeهای موجود در دیتاست (از ner_tags استخراج شد):')
    # --- BIO label space: مستقیما از خودِ ner_tags دیتاست استخراج می‌شود (حدس زده نمی‌شود) ---
    # برخلاف نسخه‌ی قبلی این Notebook، دیگر لازم نیست entityها را در متن جستجو کنیم تا BIO tag بسازیم؛
    # دیتاست از قبل هر token را با تگ BIO آماده داده (فیلد ner_tags)، پس فقط کافیست مجموعه‌ی تگ‌های
    # موجود را جمع کنیم.
    print(STRING_ENTITY_KEYS)
    bio_labels = ['O']
    for ent in STRING_ENTITY_KEYS:
        bio_labels.append(f'B-{ent}')
        bio_labels.append(f'I-{ent}')
    bio2id = {l: i for i, l in enumerate(bio_labels)}
    id2bio = {i: l for l, i in bio2id.items()}
    NUM_BIO = len(bio_labels)
    print(f'\nتعداد BIO labelها: {NUM_BIO}')
    unknown_tags = {tag for s in raw_train + raw_val + raw_test for tag in s['ner_tags']} - set(bio2id)
    assert not unknown_tags, f'تگ\u200cهای BIO ناشناخته در دیتاست پیدا شد: {unknown_tags}'
    PERSIAN_NUM_WORDS = {'یک': 1, 'یه': 1, 'ی': 1, 'دو': 2, 'سه': 3, 'چهار': 4, 'پنج': 5, 'شش': 6, 'هفت': 7, 'هشت': 8, 'نه': 9, 'ده': 10, 'یکی': 1, 'یدونه': 1, 'ید': 1, 'یدونش': 1, 'یدونشو': 1, 'دوتا': 2, 'سه\u200cتا': 3, 'چارتا': 4, 'پنجتا': 5, 'ششتا': 6, 'هفتتا': 7, 'هشتتا': 8, 'نه\u200cتا': 9, 'ده\u200cتا': 10}
    print('\nواژه\u200cهای عدد فارسی پشتیبانی\u200cشده برای تبدیل QUANTITY به عدد:', list(PERSIAN_NUM_WORDS.keys()))
    print('\n--- چند نمونه\u200cی خام برای بررسی چشمی ---')
    for _s in raw_train[:3]:
        print('\nTEXT   :', _s['text'])
        print('TOKENS :', _s['tokens'])
    # اطمینان از این‌که هیچ تگی در دیتاست خارج از bio2id نیست (وگرنه CELL 14 با KeyError متوقف می‌شود)
    # نکته درباره‌ی QUANTITY: برخلاف نسخه‌ی قبلی این Notebook، در این دیتاست QUANTITY هم یک نوع entity
    # قابل‌تشخیص با NER است (نه فقط یک قانون nearby-word) — یعنی مدل خودش یاد می‌گیرد کجای متن واژه‌ی
    # تعداد است ("دو تا"، "سه"، "3 عدد"، ...). PERSIAN_NUM_WORDS فقط برای تبدیل نهایی واژه→عدد روی همان
    # span تشخیص‌داده‌شده در CELL 24 استفاده می‌شود (نه برای پیدا کردن خودِ span).
    # نمونه‌ی سریع برای بررسی چشمی هم‌ترازی tokens/ner_tags
        print('NER    :', _s['ner_tags'])  # v21b: شکل‌های چسبیده («دوتا پیتزا میخام»؛ تست دستی دور ۱۳) + «یکی»  # v13: «ی» هم 1 است («ی پیتزا میخام» → QUANTITY=1، نه رشته‌ی "ی")
    return (
        NUM_BIO,
        NUM_INTENTS,
        NUM_SECONDARY,
        PERSIAN_NUM_WORDS,
        SECONDARY_NONE,
        STRING_ENTITY_KEYS,
        bio2id,
        id2bio,
        id2intent,
        id2secondary,
        intent2id,
        secondary2id,
    )


@app.cell
def _(
    Dataset,
    MAX_SEQ_LEN,
    MAX_SEQ_LEN_WITH_CONTEXT,
    SECONDARY_NONE,
    bio2id,
    id2bio,
    intent2id,
    raw_test,
    raw_train,
    raw_val,
    secondary2id,
    tokenizer,
):
    # CELL 14: توکنایز کردن و هم‌تراز کردن BIO labelها (که در دیتاست در سطح واژه هستند) با subword tokenها
    # دیتاست BIO را در سطح "واژه" می‌دهد (tokens/ner_tags هم‌طولند)، ولی مدل subword-tokenize می‌کند؛
    # پس باید هر تگ واژه را به اولین subword همان واژه نسبت داد و بقیه‌ی subwordهای همان واژه را -100
    # گذاشت (استاندارد HuggingFace برای token classification با pretokenized input؛ tokenizer با
    # is_split_into_words=True و word_ids() این نگاشت subword→واژه را می‌دهد).
    def align_ner_tags_to_subwords(word_ids, ner_tags):
        """word_ids: خروجی encoding.word_ids(0) — برای هر subword token، اندیس واژه\u200cی مبدأ (طبق لیست
        pretokenized ورودی) یا None برای special tokenها ([CLS]/[SEP]/[PAD]).
        فقط اولین subword هر واژه لیبل واقعی می\u200cگیرد؛ بقیه\u200cی subwordهای همان واژه -100 (در loss نادیده
        گرفته می\u200cشوند، دقیقا مثل قبل که special tokenها -100 می\u200cگرفتند)."""
        aligned = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                aligned.append(-100)
            elif wid != prev_word_id:
                tag = ner_tags[wid] if wid < len(ner_tags) else 'O'
                aligned.append(bio2id.get(tag, bio2id['O']))
            else:
                tag = ner_tags[wid] if wid < len(ner_tags) else 'O'
                sub_tag = 'I-' + tag[2:] if tag.startswith('B-') else tag if tag.startswith('I-') else 'O'  # v46: Subwords receive I- tag so the full word is learned as an entity
                aligned.append(bio2id.get(sub_tag, bio2id['O']))
            prev_word_id = wid
        return aligned

    def preprocess_example(example):
        tokens = example['tokens']
        ner_tags = example['ner_tags']
        if example.get('context'):
            context_tokens = ' '.join(example['context']).split()
            encoding = tokenizer(context_tokens, tokens, is_split_into_words=True, truncation=True, max_length=MAX_SEQ_LEN_WITH_CONTEXT, padding='max_length')
            seq_ids = encoding.sequence_ids(0)
            word_ids_cur = encoding.word_ids(0)
            bio_ids = []  # پیام‌های قبلی مکالمه (context) به‌عنوان sequence اول و پیام جاری به‌عنوان sequence دوم داده
            prev_word_id = None  # می‌شود (با [SEP] که tokenizer خودش بین دو sequence می‌گذارد). چون context به‌صورت واژه‌به‌واژه
            for seq_id, wid in zip(seq_ids, word_ids_cur):  # BIO-تگ نشده، به توکن‌های sequence اول -100 می‌دهیم (در loss نادیده گرفته می‌شوند)؛ اما برخلاف
                if seq_id != 1 or wid is None:  # نسخه‌ی قبلی این Notebook که کل NER را برای نمونه‌های context-دار خاموش می‌کرد (همه O)، اینجا
                    bio_ids.append(-100)  # چون tokens/ner_tags خودِ پیام جاری را داریم، BIO واقعی‌اش را هم نگه می‌داریم (دقیق‌تر است).
                    prev_word_id = None
                    continue
                if wid != prev_word_id:
                    tag = ner_tags[wid] if wid < len(ner_tags) else 'O'
                    bio_ids.append(bio2id.get(tag, bio2id['O']))
                else:
                    tag = ner_tags[wid] if wid < len(ner_tags) else 'O'
                    sub_tag = 'I-' + tag[2:] if tag.startswith('B-') else tag if tag.startswith('I-') else 'O'  # اندیس واژه در پیام جاری (sequence دوم) را می‌دهد؛ برای sequence اول None نیست ولی نباید استفاده شود
                    bio_ids.append(bio2id.get(sub_tag, bio2id['O']))
                prev_word_id = wid
        else:
            encoding = tokenizer(tokens, is_split_into_words=True, truncation=True, max_length=MAX_SEQ_LEN, padding='max_length')
            bio_ids = align_ner_tags_to_subwords(encoding.word_ids(0), ner_tags)
            bio_ids = bio_ids + [-100] * (MAX_SEQ_LEN_WITH_CONTEXT - len(bio_ids))
            for key in ('input_ids', 'attention_mask', 'token_type_ids'):
                if key in encoding:
                    pad_len = MAX_SEQ_LEN_WITH_CONTEXT - len(encoding[key])
                    pad_val = tokenizer.pad_token_id if key == 'input_ids' else 0
                    encoding[key] = encoding[key] + [pad_val] * pad_len
        result = {'input_ids': encoding['input_ids'], 'attention_mask': encoding['attention_mask'], 'intent_label': intent2id[example['intent']], 'secondary_label': secondary2id.get(example['secondary_intent'] or SECONDARY_NONE, secondary2id[SECONDARY_NONE]), 'clarification_label': int(bool(example['needs_clarification'])), 'ner_labels': bio_ids[:MAX_SEQ_LEN_WITH_CONTEXT]}
        if example.get('ner_ignore'):
            result['ner_labels'] = [-100] * len(result['ner_labels'])
        if 'token_type_ids' in encoding:
            result['token_type_ids'] = encoding['token_type_ids']
        return result
    _demo = next((s for s in raw_train if s.get('context')), None) or raw_train[0]
    _demo_processed = preprocess_example(_demo)
    print('--- بررسی هم\u200cترازی BIO↔subword روی یک نمونه ---')
    print('TEXT   :', _demo['text'])
    print('TOKENS :', _demo['tokens'])  # pad کردن bio_ids و بقیه‌ی فیلدها تا طول MAX_SEQ_LEN_WITH_CONTEXT، چون نمونه‌های context-دار
    print('NER    :', _demo['ner_tags'])  # طولانی‌تر توکنایز می‌شوند و همه باید در یک batch یک shape مشترک داشته باشند
    if _demo.get('context'):
        print('CONTEXT:', _demo['context'])
    _demo_subtoks = tokenizer.convert_ids_to_tokens(_demo_processed['input_ids'])
    for tok, lab in zip(_demo_subtoks, _demo_processed['ner_labels']):
        if tok == tokenizer.pad_token:
            break
        tag_str = id2bio.get(lab, '?') if lab != -100 else '-100'
        print(f'  {tok:20s} {tag_str}')
    print('\nاگر تگ\u200cهای بالا با NER نمونه هم\u200cخوانی منطقی دارند (مثلا اولین subword «استیک» → B-FOOD)، هم\u200cترازی درست کار می\u200cکند و می\u200cتوان کل دیتاست را پردازش کرد.\n')

    def build_hf_dataset(raw_examples):
        processed = [preprocess_example(ex) for ex in raw_examples]
        return Dataset.from_list(processed)
    print('در حال پردازش train...')
    train_ds = build_hf_dataset(raw_train)
    print('در حال پردازش val...')  # auto-fix (CELL 27/28): نمونه‌های اضافه‌شده‌ی خودکار BIO دستی ندارند → در loss مربوط به NER نادیده گرفته می‌شوند
    val_ds = build_hf_dataset(raw_val)
    print('در حال پردازش test...')
    test_ds = build_hf_dataset(raw_test)
    train_ds.set_format(type='torch')
    val_ds.set_format(type='torch')
    test_ds.set_format(type='torch')
    print('\nنمونه پردازش\u200cشده:')
    # --- بررسی چشمی: قبل از پردازش کل دیتاست، هم‌ترازی BIO با subwordها را روی یک نمونه‌ی واقعی چک می‌کنیم ---
    print({k: v.shape if hasattr(v, 'shape') else v for k, v in train_ds[0].items()})
    return preprocess_example, test_ds, train_ds, val_ds


@app.cell
def _(
    Counter,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    LoraConfig,
    NUM_BIO,
    NUM_INTENTS,
    NUM_SECONDARY,
    SECONDARY_NONE,
    TaskType,
    W_CLARIFICATION,
    W_INTENT,
    W_NER,
    W_SECONDARY,
    backbone,
    backbone_config,
    device,
    get_peft_model,
    intent2id,
    nn,
    raw_train,
    secondary2id,
    torch,
):
    # CELL 15: تعریف مدل چندوظیفه‌ای (Multi-task) و اعمال LoRA
    def _sqrt_inverse_weights(counts, size):
        total = sum(counts)
        active = [i for i, count in enumerate(counts) if count > 0]
        if not active:
            raise ValueError("برای یکی از سرهای طبقه‌بندی هیچ نمونه‌ای در train نیست.")
        w = [0.0] * len(counts)
        for i in active:
            w[i] = (total / (len(active) * counts[i])) ** 0.5
        mean_w = sum(w[i] for i in active) / len(active)
        return torch.tensor([x / mean_w for x in w], dtype=torch.float32, device=device)

    _intent_counts = Counter(intent2id[r["intent"]] for r in raw_train)
    _secondary_counts = Counter(secondary2id.get(r["secondary_intent"] or SECONDARY_NONE, secondary2id[SECONDARY_NONE]) for r in raw_train)
    _clarification_counts = Counter(int(bool(r["needs_clarification"])) for r in raw_train)
    INTENT_CLASS_WEIGHTS = _sqrt_inverse_weights([_intent_counts[i] for i in range(NUM_INTENTS)], NUM_INTENTS)
    SECONDARY_CLASS_WEIGHTS = _sqrt_inverse_weights([_secondary_counts[i] for i in range(NUM_SECONDARY)], NUM_SECONDARY)
    CLARIFICATION_CLASS_WEIGHTS = _sqrt_inverse_weights([_clarification_counts[i] for i in range(2)], 2)

    class MultiTaskRestaurantModel(nn.Module):
        '''backbone مشترک + 4 سر اختصاصی: intent, secondary_intent, needs_clarification, NER(BIO).'''

        def __init__(self, backbone, num_intents, num_secondary, num_bio, hidden_size, dropout=0.1):
            super().__init__()
            self.backbone = backbone
            self.dropout = nn.Dropout(dropout)
            self.intent_head = nn.Linear(hidden_size, num_intents)
            self.secondary_head = nn.Linear(hidden_size, num_secondary)
            self.clarification_head = nn.Linear(hidden_size, 2)
            self.ner_head = nn.Linear(hidden_size, num_bio)

        def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                    intent_label=None, secondary_label=None, clarification_label=None,
                    ner_labels=None, **kwargs):
            outputs = self.backbone(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
            )
            sequence_output = outputs.last_hidden_state          # (B, T, H)
            pooled = self.dropout(sequence_output[:, 0])          # [CLS] token

            intent_logits = self.intent_head(pooled)
            secondary_logits = self.secondary_head(pooled)
            clarification_logits = self.clarification_head(pooled)
            ner_logits = self.ner_head(sequence_output)

            loss = None
            if intent_label is not None:
                intent_loss = nn.functional.cross_entropy(intent_logits, intent_label, weight=INTENT_CLASS_WEIGHTS)
                secondary_loss = nn.functional.cross_entropy(secondary_logits, secondary_label, weight=SECONDARY_CLASS_WEIGHTS)
                clarification_loss = nn.functional.cross_entropy(clarification_logits, clarification_label, weight=CLARIFICATION_CLASS_WEIGHTS)
                ce_ner = nn.CrossEntropyLoss(ignore_index=-100)
                ner_loss = ce_ner(ner_logits.reshape(-1, ner_logits.size(-1)), ner_labels.reshape(-1))
                loss = (W_INTENT * intent_loss + W_SECONDARY * secondary_loss
                        + W_CLARIFICATION * clarification_loss + W_NER * ner_loss)

            return {
                "loss": loss,
                "intent_logits": intent_logits,
                "secondary_logits": secondary_logits,
                "clarification_logits": clarification_logits,
                "ner_logits": ner_logits,
            }


    base_model_for_finetune = MultiTaskRestaurantModel(
        backbone=backbone,
        num_intents=NUM_INTENTS,
        num_secondary=NUM_SECONDARY,
        num_bio=NUM_BIO,
        hidden_size=backbone_config.hidden_size,
    )

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        modules_to_save=["intent_head", "secondary_head", "clarification_head", "ner_head"],
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    model = get_peft_model(base_model_for_finetune, lora_config)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    return (model,)


@app.cell
def _(
    EVAL_BATCH_SIZE,
    EVAL_STEPS,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LOGGING_STEPS,
    NUM_EPOCHS,
    OUTPUT_DIR,
    SAVE_STEPS,
    SEED,
    TRAIN_BATCH_SIZE,
    Trainer,
    TrainingArguments,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    model,
    parse_version,
    torch,
    train_ds,
    transformers,
    val_ds,
):
    # CELL 16: پارامترهای آموزش
    use_fp16 = torch.cuda.is_available()
    use_bf16 = False  # روی GPUهای T4/V100 Colab، fp16 سازگار و کارآمد است
    if torch.cuda.is_available():  # bf16 فقط روی GPUهای جدیدتر (A100+) سازگار است؛ برای Colab پیش‌فرض False
        gpu_name = torch.cuda.get_device_name(0)
        if 'A100' in gpu_name or 'H100' in gpu_name:
            use_bf16 = True
            use_fp16 = False
        print(f'GPU: {gpu_name} -> fp16={use_fp16}, bf16={use_bf16}')
    else:
        use_fp16 = False
        print('GPU موجود نیست -> fp16/bf16 غیرفعال، آموزش روی CPU (کند خواهد بود)')
    if parse_version(transformers.__version__) >= parse_version('5.0.0'):
        warmup_kwargs = {'warmup_steps': WARMUP_RATIO}
    else:
    # نکته: از transformers v5 به بعد warmup_ratio حذف شده و warmup_steps جایگزین آن شده؛
    # warmup_steps هم عدد صحیح (تعداد step) و هم عدد اعشاری زیر ۱ (به‌عنوان نسبت/ratio،
    # دقیقاً همان کار warmup_ratio قبلی) را می‌پذیرد. کد زیر با هر دو نسخه‌ی transformers
    # (قبل و بعد از v5) کار می‌کند تا اگر باز هم پین CELL 3 دقیقاً رعایت نشد، این سلول نشکند.
        warmup_kwargs = {'warmup_ratio': WARMUP_RATIO}
    training_args = TrainingArguments(output_dir=OUTPUT_DIR, num_train_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, per_device_train_batch_size=TRAIN_BATCH_SIZE, per_device_eval_batch_size=EVAL_BATCH_SIZE, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS, **warmup_kwargs, weight_decay=WEIGHT_DECAY, eval_strategy='steps', eval_steps=EVAL_STEPS, save_strategy='steps', save_steps=SAVE_STEPS, save_total_limit=3, logging_steps=LOGGING_STEPS, load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False, fp16=use_fp16, bf16=use_bf16, gradient_checkpointing=False, report_to='none', seed=SEED, dataloader_pin_memory=torch.cuda.is_available())
    print('Training arguments آماده شد.')
    print(f'Effective batch size = {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}')
    print('\nراهنمای OOM: اگر در حین آموزش خطای CUDA Out Of Memory گرفتید:\n  1) TRAIN_BATCH_SIZE را در CELL 5 کاهش دهید (مثلا به 16 یا 8)\n  2) GRADIENT_ACCUMULATION_STEPS را افزایش دهید تا effective batch size حفظ شود\n  3) gradient_checkpointing=True را در همین سلول فعال کنید\n  4) Runtime > Restart runtime و دوباره از CELL 2 اجرا کنید')

    class MultiTaskTrainer(Trainer):
        """Trainer پیش\u200cفرض کاملا کار می\u200cکند چون مدل خودمان dict با کلید loss برمی\u200cگرداند؛
        این زیرکلاس فقط برای وضوح و امکان سفارشی\u200cسازی آینده نگه داشته شده."""
        pass
    trainer = MultiTaskTrainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds)
    print('\nTrainer آماده شد.')  # backbone کوچک است (base، نه large)؛ در صورت OOM به True تغییر دهید
    return (trainer,)


@app.cell
def _(OUTPUT_DIR, glob, os, shutil, trainer):
    # CELL 17: اجرای Fine-tuning
    # اگر آموزش قطع شد، با اجرای دوباره همین سلول از آخرین checkpoint موجود در OUTPUT_DIR ادامه می‌یابد.
    last_checkpoint = None
    if os.path.isdir(OUTPUT_DIR):
        checkpoints = glob.glob(os.path.join(OUTPUT_DIR, 'checkpoint-*'))
        if checkpoints:
            last_checkpoint = max(checkpoints, key=os.path.getmtime)
            print(f'Checkpoint موجود پیدا شد، آموزش از اینجا Resume می\u200cشود: {last_checkpoint}')
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    print('\n--- نتیجه آموزش ---')
    print(train_result.metrics)
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
    # v13: ذخیره‌ی محافظت‌شده‌ی checkpointها در Google Drive (اختیاری؛ اگر Colab وسط آموزش قطع شد،
    # CELL 17 با resume خودکار از آخرین checkpoint ادامه می‌دهد. خارج از Colab این بلوک بی‌صدا رد می‌شود.
    # دانلود/آپلود به Hugging Face به‌کلی حذف شده است — مدلی جایی آپلود نمی‌شود.)
        drive_backup_dir = '/content/drive/MyDrive/laya-restaurant-fa-checkpoints'
        os.makedirs(drive_backup_dir, exist_ok=True)
        for ckpt in glob.glob(os.path.join(OUTPUT_DIR, 'checkpoint-*')):
            dest = os.path.join(drive_backup_dir, os.path.basename(ckpt))
            if not os.path.exists(dest):
                shutil.copytree(ckpt, dest)
        print(f'\nCheckpointها در Google Drive هم ذخیره شدند: {drive_backup_dir}')
    except Exception as e:
        print(f'\nGoogle Drive در دسترس نبود — فقط checkpointهای محلی در {OUTPUT_DIR} می\u200cمانند: {e}')
    return


@app.cell
def _(
    DataLoader,
    EVAL_BATCH_SIZE,
    accuracy_score,
    device,
    id2bio,
    model,
    precision_recall_fscore_support,
    raw_test,
    raw_val,
    test_ds,
    torch,
    val_ds,
):
    model.eval()
    EVAL_SPLIT = 'val'
    assert EVAL_SPLIT in ('val', 'test'), f'EVAL_SPLIT must be val or test, got {EVAL_SPLIT!r}'
    eval_ds = val_ds if EVAL_SPLIT == 'val' else test_ds
    eval_raw = raw_val if EVAL_SPLIT == 'val' else raw_test
    print(f'ارزیابی روی split: {EVAL_SPLIT} ({len(eval_raw)} سطر)')
    val_loader = DataLoader(eval_ds, batch_size=EVAL_BATCH_SIZE)
    all_intent_preds, all_intent_true = ([], [])
    _all_secondary_preds, _all_secondary_true = ([], [])
    _all_clarif_preds, _all_clarif_true = ([], [])
    _all_ner_preds, _all_ner_true = ([], [])
    with torch.no_grad():
        for _batch in val_loader:
            _batch_on_device = {k: v.to(device) for k, v in _batch.items()}
            _outputs = model(**_batch_on_device)
            _intent_pred = _outputs['intent_logits'].argmax(-1).cpu().numpy()
            _secondary_pred = _outputs['secondary_logits'].argmax(-1).cpu().numpy()
            _clarif_pred = _outputs['clarification_logits'].argmax(-1).cpu().numpy()
            _ner_pred = _outputs['ner_logits'].argmax(-1).cpu().numpy()
            all_intent_preds.extend(_intent_pred.tolist())
            all_intent_true.extend(_batch['intent_label'].numpy().tolist())
            _all_secondary_preds.extend(_secondary_pred.tolist())
            _all_secondary_true.extend(_batch['secondary_label'].numpy().tolist())
            _all_clarif_preds.extend(_clarif_pred.tolist())
            _all_clarif_true.extend(_batch['clarification_label'].numpy().tolist())
            _ner_true_batch = _batch['ner_labels'].numpy()
            for _i in range(len(_ner_pred)):
                _seq_pred, _seq_true = ([], [])
                for _p, _t in zip(_ner_pred[_i], _ner_true_batch[_i]):
                    if _t == -100:
                        continue
                    _seq_pred.append(id2bio[_p])
                    _seq_true.append(id2bio[_t])
                _all_ner_preds.append(_seq_pred)
                _all_ner_true.append(_seq_true)
    acc = accuracy_score(all_intent_true, all_intent_preds)
    _p_macro, _r_macro, f1_macro, _ = precision_recall_fscore_support(all_intent_true, all_intent_preds, average='macro', zero_division=0)
    _p_weighted, _r_weighted, f1_weighted, _ = precision_recall_fscore_support(all_intent_true, all_intent_preds, average='weighted', zero_division=0)
    print('=== Intent classification ===')
    print(f'Accuracy:      {acc:.4f}')
    print(f'Macro    P/R/F1: {_p_macro:.4f} / {_r_macro:.4f} / {f1_macro:.4f}')
    print(f'Weighted P/R/F1: {_p_weighted:.4f} / {_r_weighted:.4f} / {f1_weighted:.4f}')
    _sec_acc = accuracy_score(_all_secondary_true, _all_secondary_preds)
    print(f'\n=== Secondary intent ===\nAccuracy: {_sec_acc:.4f}')
    _clarif_acc = accuracy_score(_all_clarif_true, _all_clarif_preds)
    _, _, clarif_f1, _ = precision_recall_fscore_support(_all_clarif_true, _all_clarif_preds, average='binary', zero_division=0)
    print(f'\n=== needs_clarification ===\nAccuracy: {_clarif_acc:.4f}   F1: {clarif_f1:.4f}')

    def _get_entities_bio(seq):
        entities, start, etype = ([], None, None)
        for i, tag in enumerate(seq):
            if tag.startswith('B-'):
                if etype is not None:
                    entities.append((etype, start, i - 1))
                etype, start = (tag[2:], i)
            elif tag.startswith('I-') and etype == tag[2:]:
                continue
            else:
                if etype is not None:
                    entities.append((etype, start, i - 1))
                etype = None
        if etype is not None:
            entities.append((etype, start, len(seq) - 1))
        return entities

    def _entity_level_report(y_true, y_pred):
        true_ents, pred_ents = (set(), set())
        for i, (t_seq, p_seq) in enumerate(zip(y_true, y_pred)):
            true_ents = true_ents | {(i, *e) for e in _get_entities_bio(t_seq)}
            pred_ents = pred_ents | {(i, *e) for e in _get_entities_bio(p_seq)}

        def _prf(t_set, p_set):
            tp = len(t_set & p_set)
            fp = len(p_set - t_set)
            fn = len(t_set - p_set)
            pr = tp / (tp + fp) if tp + fp else 0.0
            rc = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
            return (pr, rc, f1, len(t_set))
        overall_p, overall_r, overall_f1, support = _prf(true_ents, pred_ents)
        types = sorted({e[1] for e in true_ents} | {e[1] for e in pred_ents})
        lines = [f"{'type':<20s}{'precision':>10s}{'recall':>10s}{'f1-score':>10s}{'support':>10s}"]
        for etype in types:
            t_e = {e for e in true_ents if e[1] == etype}
            p_e = {e for e in pred_ents if e[1] == etype}
            pr, rc, f1, sup = _prf(t_e, p_e)
            lines.append(f'{etype:<20s}{pr:10.4f}{rc:10.4f}{f1:10.4f}{sup:10d}')
        lines.append(f"{'micro avg':<20s}{overall_p:10.4f}{overall_r:10.4f}{overall_f1:10.4f}{support:10d}")
        return (overall_f1, '\n'.join(lines))
    print('\n=== Entity extraction (NER, entity-level) ===')
    _ner_f1, _ner_report = _entity_level_report(_all_ner_true, _all_ner_preds)
    print(f'Entity-level F1: {_ner_f1:.4f}')
    print(_ner_report)
    return (
        EVAL_SPLIT,
        acc,
        all_intent_preds,
        all_intent_true,
        clarif_f1,
        eval_raw,
        f1_macro,
        f1_weighted,
    )


@app.cell
def _(
    ALL_INTENTS,
    all_intent_preds,
    all_intent_true,
    classification_report,
    id2intent,
):
    # CELL 19: Classification report تفصیلی برای هر Intent
    intent_names_true = [id2intent[i] for i in all_intent_true]
    intent_names_pred = [id2intent[i] for i in all_intent_preds]
    report_str = classification_report(intent_names_true, intent_names_pred, labels=ALL_INTENTS, zero_division=0)
    print(report_str)
    report_dict = classification_report(intent_names_true, intent_names_pred, labels=ALL_INTENTS, output_dict=True, zero_division=0)
    weak_intents = sorted(((intent, report_dict[intent]['f1-score'], int(report_dict[intent]['support'])) for intent in ALL_INTENTS if report_dict[intent]['support'] > 0), key=lambda x: x[1])[:10]
    print('\n--- 10 intent با ضعیف\u200cترین F1 ---')
    print(f"{'intent':30s} {'F1':>6} {'support':>8}")
    # جدول intentهایی با عملکرد ضعیف (F1 پایین)
    for _intent, f1, support in weak_intents:
        print(f'{_intent:30s} {f1:>6.2f} {support:>8}')
    return intent_names_pred, intent_names_true


@app.cell
def _(ALL_INTENTS, confusion_matrix, intent_names_pred, intent_names_true):
    # CELL 20: Confusion Matrix
    import matplotlib.pyplot as plt
    cm = confusion_matrix(intent_names_true, intent_names_pred, labels=ALL_INTENTS)
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(ALL_INTENTS)))
    ax.set_yticks(range(len(ALL_INTENTS)))
    ax.set_xticklabels(ALL_INTENTS, rotation=90, fontsize=7)
    ax.set_yticklabels(ALL_INTENTS, fontsize=7)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix — Intent Classification')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()
    print('ذخیره شد: confusion_matrix.png')
    print('\n--- بیشترین جفت\u200cهای خطای Intent (True -> Predicted) ---')
    errors = []
    for _i, true_i in enumerate(ALL_INTENTS):
    # جفت‌های intent که بیشترین اشتباه بین‌شان رخ داده (غیر از قطر اصلی)
        for j, pred_j in enumerate(ALL_INTENTS):
            if _i != j and cm[_i][j] > 0:
                errors.append((cm[_i][j], true_i, pred_j))
    errors.sort(reverse=True)
    for count, true_i, pred_j in errors[:15]:
        print(f'{true_i:25s} -> {pred_j:25s} : {count} بار')
    return


@app.cell
def _(
    EVAL_SPLIT,
    all_intent_preds,
    all_intent_true,
    eval_raw,
    id2intent,
    raw_test,
    raw_train,
    raw_val,
):
    # CELL 21: تحلیل خطا — حداقل ۳۰ prediction اشتباه
    action_map = {}  # از intent -> action، برای نمایش expected/predicted action
    for _s in raw_train + raw_val + raw_test:
        action_map[_s['intent']] = _s['action']
    wrong_indices = [i for i in range(len(all_intent_true)) if all_intent_true[i] != all_intent_preds[i]]
    print(f'تعداد کل پیش\u200cبینی\u200cهای اشتباه (intent) در {EVAL_SPLIT}: {len(wrong_indices)} از {len(all_intent_true)}\n')
    sample_wrong = wrong_indices[:30] if len(wrong_indices) >= 30 else wrong_indices
    if len(sample_wrong) < 30:
        print(f'⚠️ فقط {len(sample_wrong)} نمونه اشتباه در test وجود دارد (کمتر از ۳۰).\n')
    for idx in sample_wrong:
        true_intent = id2intent[all_intent_true[idx]]
        pred_intent = id2intent[all_intent_preds[idx]]
        _row = eval_raw[idx]
        _text = _row['text']
        _ctx = _row.get('context') or [h.get('text', '') for h in _row.get('history') or []]
        print('TEXT:')  # همان split که مدل روی آن پیش‌بینی کرد
        print(f'  {_text}\n')
        if _ctx:
            print('PRECEDING BOT MESSAGE:')
            print(f'  {_ctx[-1]}\n')
        print('TRUE:')
        print(f'  {true_intent}\n')
        print('PREDICTED:')
        print(f'  {pred_intent}\n')
        print('EXPECTED ACTION:')
        print(f"  {action_map.get(true_intent, '?')}\n")
        print('PREDICTED ACTION:')
        print(f"  {action_map.get(pred_intent, '?')}\n")
        print('-' * 50)
    return (action_map,)


@app.cell
def _(
    BASE_MODEL_NAME,
    FINAL_MODEL_DIR,
    MAX_SEQ_LEN,
    MAX_SEQ_LEN_WITH_CONTEXT,
    PERSIAN_NUM_WORDS,
    STRING_ENTITY_KEYS,
    action_map,
    bio2id,
    intent2id,
    json,
    model,
    os,
    secondary2id,
    tokenizer,
):
    # CELL 22: ذخیره مدل نهایی
    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    model.save_pretrained(FINAL_MODEL_DIR)
    # 1) آداپتور LoRA + سرهای اختصاصی (که با modules_to_save کنار adapter ذخیره می‌شوند)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    metadata = {'base_model': BASE_MODEL_NAME, 'intent2id': intent2id, 'secondary2id': secondary2id, 'bio2id': bio2id, 'max_seq_len': MAX_SEQ_LEN, 'max_seq_len_with_context': MAX_SEQ_LEN_WITH_CONTEXT, 'action_map': action_map, 'string_entity_keys': STRING_ENTITY_KEYS, 'persian_num_words': PERSIAN_NUM_WORDS}
    # 2) tokenizer
    with open(os.path.join(FINAL_MODEL_DIR, 'task_metadata.json'), 'w', encoding='utf-8') as _f:
        json.dump(metadata, _f, ensure_ascii=False, indent=2)
    # 3) label mappings و متادیتای لازم برای inference
    print(f'مدل، tokenizer و متادیتا در {FINAL_MODEL_DIR} ذخیره شدند:')
    for fname in os.listdir(FINAL_MODEL_DIR):
        print(' -', fname)
    return


@app.cell
def _(
    ALL_INTENTS,
    BASE_MODEL_NAME,
    FINAL_MODEL_DIR,
    LORA_ALPHA,
    LORA_R,
    RAW_DATASET_FILENAME,
    acc,
    clarif_f1,
    f1_macro,
    f1_weighted,
    os,
    raw_test,
    raw_train,
    raw_val,
    shutil,
):
    model_card = f'---\nlanguage: fa\nlicense: apache-2.0\ntags:\n  - persian\n  - farsi\n  - intent-classification\n  - named-entity-recognition\n  - restaurant\n  - chatbot\n  - peft\n  - lora\nbase_model: {BASE_MODEL_NAME}\n---\n\n# مدل فارسی چت\u200cبات سفارش رستوران (Intent + Entity Extraction)\n\n## Base model\n`{BASE_MODEL_NAME}` (ParsBERT) + آداپتورهای LoRA + ۴ سر اختصاصی (intent / secondary-intent / needs_clarification / NER).\n\n## Dataset\nدیتاست فارسی رستوران («{RAW_DATASET_FILENAME}»)، شامل {len(raw_train) + len(raw_val) + len(raw_test)} نمونه، با annotation در سطح token برای NER (BIO scheme).\n\n## Language\nفارسی (فرمال + محاوره\u200cای)\n\n## Task\nتحلیل پیام کاربر رستوران آنلاین و تولید خروجی ساختاریافته:\n`{{"intent", "secondary_intent", "entities", "action", "needs_clarification"}}`\n\n## Intent list\n{ALL_INTENTS}\n\n## Training method\nLoRA (r={LORA_R}, alpha={LORA_ALPHA}) روی backbone + fine-tune کامل ۴ سر اختصاصی، با Trainer استاندارد HuggingFace.\n\n## Dataset size / split\ntrain: {len(raw_train)} — val: {len(raw_val)} — test: {len(raw_test)} (stratified 80/10/10)\n\n## Evaluation results (روی validation set (pre-KFold model))\n- Intent accuracy: {acc:.4f}\n- Intent macro F1: {f1_macro:.4f}\n- Intent weighted F1: {f1_weighted:.4f}\n- needs_clarification F1: {clarif_f1:.4f}\n\n## Example input/output\nورودی: «دو تا پیتزا پپرونی میخوام ولی یکی\u200cشو حذف کن»\nخروجی: `{{"intent": "decrease_quantity", "entities": {{"food": "پیتزا پپرونی", "quantity_change": 1}}, "action": "decrease"}}`\n\n## Limitations\n- روی دامنه محدود سفارش رستوران فارسی train شده؛ به دامنه\u200cهای دیگر تعمیم نمی\u200cیابد.\n- استخراج quantity: خودِ span با NER تشخیص داده می\u200cشود (نوع entity: QUANTITY)؛ فقط تبدیل نهایی واژه→عدد («دو تا» → 2) با یک قانون ساده مبتنی بر واژه\u200cهای عدد فارسی انجام می\u200cشود (CELL 24).\n- دیتاست به\u200cصورت مصنوعی (synthetic) تولید شده، نه از مکالمات واقعی کاربران.\n\n## Intended use\nراهنمای NLU برای یک سیستم سفارش رستوران فارسی؛ خروجی آن باید توسط منطق backend مصرف شود، نه به\u200cعنوان پاسخ مستقیم به کاربر.\n'
    with open(os.path.join(FINAL_MODEL_DIR, 'README.md'), 'w', encoding='utf-8') as _f:
        _f.write(model_card)
    if True:
        print('مدل فقط لوکال ذخیره و دانلود می\u200cشود — هیچ\u200cچیز جایی پابلیک نمی\u200cشود.')
        zip_base = 'parsbert-restaurant-fa-model'
        zip_path = shutil.make_archive(zip_base, 'zip', FINAL_MODEL_DIR)
        print(f'فایل zip ساخته شد: {zip_path}')
        try:
            from google.colab import files
            files.download(zip_path)
            print('دانلود شروع شد (اگر خودکار دانلود نشد، از پنل Files سمت چپ روی همین فایل کلیک راست کنید و Download بزنید).')
        except ImportError:
            print(f'خارج از Colab هستید — فایل zip در مسیر {zip_path} روی دیسک ذخیره شد.')
    return


@app.cell
def _(
    FA_CHAR_MAP,
    MAX_SEQ_LEN,
    MAX_SEQ_LEN_WITH_CONTEXT,
    PERSIAN_NUM_WORDS,
    SECONDARY_NONE,
    action_map,
    defaultdict,
    device,
    id2bio,
    id2intent,
    id2secondary,
    json,
    model,
    re,
    tokenizer,
    torch,
):
    # CELL 24: رابط تست تعاملی
    CONFIDENCE_THRESHOLD = 0.55  # اگر بیشترین احتمال intent زیر این مقدار باشد، needs_clarification=True فرض می‌شود
    PERSIAN_DIGITS_TO_LATIN = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")


    QUANTITY_MARKERS = {"تا", "عدد", "دونه", "پرس", "نمره"}
    ORDER_INTENTS_FOR_NINE = {"place_order", "add_item", "modify_item", "increase_quantity"}
    NEGATIVE_INTENTS_FOR_NINE = {"deny", "reject_order_confirmation", "reject_address_reuse", "reject_cancel_order", "no_more_items"}


    def _q_norm(text):
        return re.sub(r"\s+", " ", (text or "").translate(PERSIAN_DIGITS_TO_LATIN)
                      .replace("ك", "ک").replace("ي", "ی").replace("\u200c", " ")).strip()


    def _has_nine_marker(span_text, full_text=None, span_end=None):
        if re.search(r"(?<!\w)نه\s*(?:تا|عدد|دونه|پرس|نمره)(?!\w)", _q_norm(span_text)):
            return True
        tail = _q_norm(full_text[span_end:]) if full_text is not None and span_end is not None else ""
        return bool(re.match(r"^(?:تا|عدد|دونه|پرس|نمره)(?!\w)", tail))


    def _confirmation_context(context):
        ctx = _q_norm(" ".join(str(x) for x in (context or [])))
        topic = re.search(r"تایید|تأیید|ثبت\s+نهایی|آدرس\s+قبلی|لغو\s+سفارش|چیز\s+دیگه|مورد\s+دیگه", ctx)
        asks = "؟" in ctx or "?" in ctx or bool(re.search(r"بله\s+یا\s+(?:خیر|نه)", ctx))
        return bool(topic and asks)


    def _nine_is_negative(text, intent=None, context=None):
        norm = _q_norm(text)
        if not re.search(r"(?<!\w)نه(?!\w)", norm):
            return False
        if re.search(r"(?<!\w)نه\s*(?:تا|عدد|دونه|پرس|نمره)(?!\w)", norm):
            return False
        if re.search(r"(?<!\w)9\s*(?:تا|عدد|دونه|پرس)(?!\w)", norm):
            return False
        if intent in NEGATIVE_INTENTS_FOR_NINE or re.search(r"^\s*نه\s*[،,؛:]", text or ""):
            return True
        if re.search(r"نمی\s*خ|نخوا|نیار|نیاور", norm):
            return True
        return _confirmation_context(context) and bool(re.match(r"^\s*(?:نه|نخیر|خیر)(?:\s|$)", text or ""))


    def _clear_nine_order(text, intent, context, has_food):
        if intent not in ORDER_INTENTS_FOR_NINE or not has_food or _nine_is_negative(text, intent, context):
            return False
        return bool(re.search(r"می\s*خوا|بگیر|بده|بدید|بزن|اضافه\s*کن|بذار|بزار|سفارش\s*می", _q_norm(text)))


    def _ambiguous_nine(text, intent, context, has_food):
        norm = _q_norm(text)
        if not has_food or not re.match(r"^نه(?:\s|$)", norm) or _has_nine_marker(norm):
            return False
        if _confirmation_context(context) and re.match(r"^\s*(?:نه|نخیر|خیر)(?:\s|$)", text or ""):
            return True
        if _nine_is_negative(text, intent, context):
            return False
        return not _clear_nine_order(text, intent, context, has_food)


    def parse_quantity_text(span_text, full_text=None, span_end=None, intent=None, context=None, has_food=False):
        # A bare «نه» is nine only with an explicit count marker or a clear order phrase.
        cleaned = _q_norm(span_text)
        m = re.search(r"\d+", cleaned)
        if m:
            return int(m.group())
        words = re.findall(r"[\u0600-\u06FF]+", cleaned)
        if any(w.replace("\u200c", " ").strip() == "نه" for w in words):
            if _has_nine_marker(span_text, full_text, span_end):
                return 9
            if full_text is not None and _clear_nine_order(full_text, intent, context, has_food):
                return 9
            return None
        for word in words:
            if word in PERSIAN_NUM_WORDS:
                return PERSIAN_NUM_WORDS[word]
            for suffix in ("دونه", "تا", "نمره", "عدد"):
                if word.endswith(suffix) and len(word) > len(suffix):
                    stem = word[:-len(suffix)]
                    if stem in PERSIAN_NUM_WORDS and stem != "نه":
                        return PERSIAN_NUM_WORDS[stem]
                    if stem.isdigit():
                        return int(stem.translate(PERSIAN_DIGITS_TO_LATIN))
        return None


    def extract_quantity_near_span(text, span_start, span_end, intent=None, context=None):
        before = text[:span_start]
        words = re.findall(r"[\u0600-\u06FF]+|\d+", before)
        for i in range(len(words)-1, max(-1,len(words)-5), -1):
            word = words[i].replace("\u200c"," ").strip()
            if word == "نه":
                marker = i+1 < len(words) and words[i+1] in QUANTITY_MARKERS
                if marker or _clear_nine_order(text,intent,context,has_food=True):
                    return 9
                continue
            if word.isdigit():
                return int(word)
            if word in PERSIAN_NUM_WORDS:
                return PERSIAN_NUM_WORDS[word]
        return None


    def decode_bio_to_entities(text, offset_mapping, bio_pred_ids, intent=None, context=None):
        '''offset_mapping/bio_pred_ids باید فقط مربوط به توکن‌های پیام جاری باشند (نه توکن‌های context؛
        predict_message قبل از فراخوانی این تابع آن‌ها را فیلتر می‌کند).'''
        entities = defaultdict(list)   # v13: لیست (start, end, text) — برای merge اسپن‌های چسبیده
        current_type, current_start, current_end = None, None, None

        def flush():
            if current_type is not None:
                entities[current_type.lower()].append((current_start, current_end, text[current_start:current_end]))

        for (start, end), label_id in zip(offset_mapping, bio_pred_ids):
            if start == end:
                continue
            label = id2bio.get(label_id, "O")
            if label.startswith("B-"):
                flush()
                current_type = label[2:]
                current_start, current_end = start, end
            elif label.startswith("I-") and current_type == label[2:]:
                current_end = end
            else:
                flush()
                current_type = None
        flush()

        # v13: merge اسپن‌های هم‌نوعی که چسبیده‌اند (مثل «توشابه» که subwordها آن را
        # «توش» + «ابه» جدا تشخیص داده‌اند) — فاصله‌ی بین دو اسپن <= 1 کاراکتر → یکی می‌شوند
        merged = defaultdict(list)
        for etype, spans in entities.items():
            for (s, e, txt) in spans:
                if merged[etype] and s - merged[etype][-1][1] <= 1:
                    ps, pe = merged[etype][-1][0], merged[etype][-1][1]
                    merged[etype][-1] = (ps, e, text[ps:e])
                else:
                    merged[etype].append((s, e, txt))
        entities = merged

        result = {}
        for k, v in entities.items():
            if k == "quantity":
                has_food = any(kind in merged for kind in ("food", "drink", "dessert"))
                parsed = []
                for (start, end, span_text) in v:
                    value = parse_quantity_text(span_text, text, end, intent, context, has_food)
                    if value is not None:
                        parsed.append(value)
                    elif not re.search(r"(?<!\w)نه(?!\w)", _q_norm(span_text)):
                        parsed.append(span_text)
                if parsed:
                    result[k] = parsed[0] if len(parsed) == 1 else parsed
            else:
                texts_k = [t for (_, _, t) in v]
                result[k] = texts_k[0] if len(texts_k) == 1 else texts_k

        # fallback: اگر NER هیچ QUANTITY ای پیدا نکرد، نزدیک اولین FOOD/DRINK/DESSERT دنبال واژه‌ی عدد بگرد
        if "quantity" not in result:
            for ent_type in ("food", "drink", "dessert"):
                if ent_type not in result:
                    continue
                span_vals = result[ent_type] if isinstance(result[ent_type], list) else [result[ent_type]]
                for span_text in span_vals:
                    span_start = text.find(span_text)
                    if span_start != -1:
                        qty = extract_quantity_near_span(text, span_start, span_start + len(span_text), intent, context)
                        if qty is not None:
                            result["quantity"] = qty
                            break
                if "quantity" in result:
                    break

        return result


    def predict_message(text, context=None):
        # v46: Clean emojis & control chars so embeddings match training preprocessing
        try:
            text = re.sub(r"[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\uFE0F\u2B50\u2B55\u2705\u274C\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]", "", text)
            text = re.sub(r"[ \t\u00A0]+", " ", text).strip()
        except Exception:
            pass
        # v13: ورودی کاربر هم نرمال‌سازی ك/ی می‌شود (کیبورد عربی → فارسی) تا با دیتاست نرمال‌شده هم‌خوان بماند
        try:
            text = text.translate(FA_CHAR_MAP)
        except Exception:
            pass
        model.eval()
        if context:
            ctx_text = " ".join(context)
            encoding = tokenizer(
                ctx_text, text, truncation=True, max_length=MAX_SEQ_LEN_WITH_CONTEXT,
                padding="max_length", return_offsets_mapping=True, return_tensors="pt",
            )
            # offset_mapping هر sequence نسبت به رشته‌ی خودش است، نه رشته‌ی الحاق‌شده؛ پس توکن‌های
            # context را قبل از decode کنار می‌گذاریم (فقط sequence دوم = پیام جاری معتبر است)
            seq_ids = encoding.sequence_ids(0)
            keep_idx = [i for i, sid in enumerate(seq_ids) if sid == 1]
        else:
            encoding = tokenizer(
                text, truncation=True, max_length=MAX_SEQ_LEN, padding="max_length",
                return_offsets_mapping=True, return_tensors="pt",
            )
            keep_idx = list(range(encoding["input_ids"].shape[1]))

        offsets_raw = encoding["offset_mapping"][0].tolist()

        inputs = {k: v.to(device) for k, v in encoding.items() if k in ("input_ids", "attention_mask", "token_type_ids")}

        with torch.no_grad():
            outputs = model(**inputs)

        intent_probs = torch.softmax(outputs["intent_logits"], dim=-1)[0]
        intent_id = int(intent_probs.argmax())
        intent_conf = float(intent_probs[intent_id])
        intent_name = id2intent[intent_id]

        secondary_id = int(outputs["secondary_logits"].argmax(-1)[0])
        secondary_name = id2secondary[secondary_id]
        secondary_out = None if secondary_name == SECONDARY_NONE else secondary_name

        clarif_pred = int(outputs["clarification_logits"].argmax(-1)[0])
        needs_clarification = bool(clarif_pred) or (intent_conf < CONFIDENCE_THRESHOLD)

        bio_pred_ids_all = outputs["ner_logits"].argmax(-1)[0].cpu().tolist()
        offsets_for_decoding = [offsets_raw[i] for i in keep_idx]
        bio_pred_ids = [bio_pred_ids_all[i] for i in keep_idx]
        entities = decode_bio_to_entities(text, offsets_for_decoding, bio_pred_ids, intent_name, context)
        has_food = any(k in entities for k in ("food", "drink", "dessert"))
        needs_clarification = needs_clarification or _ambiguous_nine(text, intent_name, context, has_food)

        return {
            "intent": intent_name,
            "secondary_intent": secondary_out,
            "needs_clarification": needs_clarification,
            "entities": entities,
            "action": action_map.get(intent_name, "unknown"),
            "_intent_confidence": round(intent_conf, 3),   # فقط برای دیباگ، در خروجی نهایی محصول لازم نیست
        }


    print("رابط تست تعاملی آماده است.")
    print("برای تست دستی، در همین سلول مقدار متغیر _msg را عوض کنید و سلول را دوباره اجرا کنید (یا از CELL 26 استفاده کنید):\n")

    _msg = "دو تا پیتزا پپرونی میخوام"   # ← پیام تست را اینجا عوض کنید
    print("پیام مشتری:", _msg)
    print(json.dumps(predict_message(_msg), ensure_ascii=False, indent=2))
    return (predict_message,)


@app.cell
def _(json, predict_message):
    # CELL 25: تست روی چند مکالمه نمونه رستوران
    example_messages = [
        "سلام وقت بخیر",
        "دو تا پیتزا پپرونی میخوام",
        "یکی از پیتزاها رو حذف کن",
        "یه نوشابه هم اضافه کن",
        "سفارشم کجاست؟",
        "غذا سرد به دستم رسید",
        "سفارشم رو کنسل کنید",
        "کد تخفیف دارید؟",
        "تا چند بازید؟",
    ]

    for msg in example_messages:
        result = predict_message(msg)
        print("TEXT:", msg)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("=" * 50)

    # --- نمونه‌ای از استفاده با context (مکالمه دو مرحله‌ای) ---
    print("\n--- مثال context-دار ---")
    prev_msg = "دو تا پیتزا پپرونی و یه نوشابه میخوام"
    cur_msg = "نوشابه رو حذف کن"
    result = predict_message(cur_msg, context=[prev_msg])
    print("PREV:", prev_msg)
    print("CUR: ", cur_msg)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return


@app.cell
def _(predict_message):
    REGRESSION_CASES = [('سااام', 'greeting'), ('همه رو حذف کن', 'remove_all_cart'), ('ی پپرونی بده با پنیر اضافه بدون گوجه', 'place_order', None, {'food': 'پپرونی', 'remove_ingredient': 'گوجه'}), ('آدرسم گلستان 15', 'address_change'), ('کی برام می\u200cفرستی؟', 'ask_delivery_time'), ('چ غداهایی دارین؟', 'ask_menu'), ('یه برگر بیکن از سفارش کم کن', 'decrease_quantity'), ('یک براونی هم اضافه کن', 'add_item'), ('لیموناد بگیرید', 'add_item'), ('املت سوسیس رو ۱ تا کن', 'increase_quantity'), ('بفرست', 'confirm_order'), ('ترکوندی', 'thanks', ['سفارشتون با موفقیت ثبت شد.']), ('باثلام', 'greeting'), ('پفثت', 'affirm'), ('مثجد کجاست؟', 'out_of_scope'), ('ثاعت كاری رثتوران چندا؟', 'ask_working_hours'), ('مغررات رثتوران چیح؟', 'ask_restaurant_rules'), ('بفطث', 'affirm'), ('واقعا ممغن', 'thanks'), ('کارت به کارت', 'select_payment_method'), ('گلثتان 15', 'address_change'), ('جوجه کباب بدون استخوان', 'place_order'), ('ثاهت کاریتون چندا؟', 'ask_working_hours'), ('چیذبرگر', 'place_order'), ('شمارح تماث شعبا شعبه فردوث چندح؟', 'ask_phone'), ('آدرث دقیغ رثتوران چیح؟ دوثت عزیذ', 'ask_address'), ('برنامح\u200cی تلویذیون امشب چیح؟، لطفا', 'out_of_scope'), ('صبه\u200cها هم باذید؟', 'ask_working_hours'), ('یه تقییری بده', 'modify_item'), ('ممنغ رفیق', 'thanks'), ('ممغن رفیق', 'thanks'), ('ثلامت باشید 😊', 'goodbye'), ('سلام', 'greeting'), ('چ خبر', 'small_talk'), ('چطوری', 'small_talk'), ('خوبی؟', 'small_talk'), ('چیکار میکنی', 'small_talk'), ('خدافظ', 'goodbye'), ('خدافظظ', 'goodbye'), ('خددافط', 'goodbye'), ('بله', 'affirm'), ('بله', 'want_more_items', ['چیز دیگه\u200cای میل دارید؟ اگر همین\u200cها بود بگید ثبت کنم.']), ('بله', 'confirm_address', ['به همون آدرس قبلی، براتون بفرستم؟']), ('بله', 'confirm_order', ['سفارش رو تایید می\u200cکنید؟']), ('بله', 'confirm_cancel_order', ['لغو سفارش رو تایید می\u200cکنید؟']), ('باشه', 'affirm'), ('باشه', 'confirm_order', ['سفارش رو ثبت نهایی کنم؟']), ('اوکی', 'affirm'), ('اوکی', 'confirm_order', ['سفارش رو ثبت نهایی کنم؟']), ('نه', 'deny'), ('نخیر', 'deny'), ('خیر', 'deny'), ('ی پیتزا میخام', 'place_order', None, {'quantity': 1, 'food': 'پیتزا'}), ('دو تا پیتزا میخام', 'place_order', None, {'quantity': 2}), ('سه تا پپرونی', 'place_order', None, {'quantity': 3}), ('ی توشابه', 'place_order', None, {'drink': 'توشابه'}), ('ی پیتزا داری', 'ask_availability'), ('نخیر بابا', 'deny'), ('فثنجان', 'place_order'), ('ممنغ', 'thanks'), ('موذیک دوثت داری؟ دوثت عذیز', 'small_talk'), ('سبتش کن', 'confirm_order'), ('امتیاذ مسبت چی دارین؟', 'ask_discount'), ('ثلامت باشید 😊', 'goodbye'), ('گلثتان 15', 'address_change'), ('ی پیتزا داری', 'ask_availability'), ('عدثی', 'place_order'), ('حلیمم', 'place_order', ['چه غذایی برایتان ثبت کنم؟']), ('حوممم', 'unclear'), ('نیت خیر دارم برات', 'small_talk'), ('نیت خیر دارم برات 🌹', 'small_talk'), ('اگه غذا اشکال داشت چیکار کنم؟', 'ask_restaurant_rules'), ('ننه', 'deny'), ('چطوری؟', 'small_talk'), ('ممنغ از لطفت', 'thanks'), ('یه چیز گرم بده', 'place_order'), ('ی پیتزا داری', 'ask_availability'), ('گلثتان 15', 'address_change'), ('نخیر بابا', 'deny'), ('ممنغ', 'thanks'), ('فثنجان', 'place_order'), ('ثلامت باشید 😊', 'goodbye'), ('حمین سبد رو نحایی کن', 'confirm_order'), ('نیت خیر دارم برات 🍽️', 'small_talk'), ('غوانین رسترون رو بگید', 'ask_restaurant_rules'), ('سفارشم رو تقییر بده', 'modify_item'), ('ننه', 'deny'), ('قیمت بورث امروذ چنده؟ 🍽️', 'out_of_scope'), ('اوکی اوکی', 'affirm'), ('نیت خیر دارم برات', 'small_talk'), ('نیت خیر دارم برات 🌹', 'small_talk'), ('مرجوعی رو چطور پس بگیرم؟', 'ask_restaurant_rules'), ('مرجوعی تا چند دقیغه بعد از تهویله؟', 'ask_restaurant_rules'), ('غوانینتون چیه؟', 'ask_restaurant_rules'), ('همینه، ثفارش رو نحایی کن', 'confirm_order'), ('صبحانه چی خوردی؟', 'small_talk'), ('کتاب چی می\u200cخونی؟', 'small_talk'), ('یه قهوه چه می\u200cکنه؟', 'small_talk'), ('سپاثگزارم', 'thanks'), ('شمارا تماث رثتوران چندح؟', 'ask_phone'), ('نیت خیر دارم برات 🌹', 'small_talk'), ('چح\u200cکاره\u200cای', 'unclear'), ('خدا خیرتون بده', 'thanks'), ('ذیتون لطفا', 'place_order'), ('سپاثگزارم', 'thanks'), ('ممنغ', 'thanks'), ('قربون مهربونیت برم', 'thanks'), ('نغدی', 'select_payment_method'), ('ثلامت بایذ', 'goodbye'), ('خسته نباشی', 'small_talk'), ('چند روز مثافرت رفتم', 'small_talk'), ('چند روز مسافرت رفتم', 'small_talk'), ('مسافرت بودم', 'small_talk'), ('چند روز کسی نبودم', 'small_talk'), ('از کجا می\u200cتونم سفارش حضوری بگیرم؟ 🙏', 'ask_address'), ('از کجا سفارش حضوری بگیرم؟', 'ask_address'), ('آدرس رستوران', 'ask_address'), ('ادرستون', 'ask_address'), ('دوتا پیتزا پپرونی میخام', 'place_order', None, {'quantity': 2}), ('یدونه پیتزا میخام', 'place_order', None, {'quantity': 1}), ('یکی از پیتزا پپرونی میخاستم', 'place_order', None, {'quantity': 1}), ('دوتا پیتزا میخام', 'place_order', None, {'quantity': 2, 'food': 'پیتزا'}), ('فردا چخبره؟', 'small_talk'), ('خدا پدر مادرت رو بیامرزه', 'thanks'), ('خدا پدر مادرت رو بیامرزه 🙏', 'thanks'), ('عید پیش\u200cاپیش مبارک', 'small_talk'), ('سفارشم به آدرس اشتباح رفت', 'complaint_order'), ('ثفارشم به آدرس اشتباح رفت', 'complaint_order'), ('بذار یکی بشه', 'decrease_quantity', ['دو پیتزا در سفارش شماست.']), ('برنامح\u200cی تلویذیون امشب چیح؟، لطفا', 'out_of_scope'), ('تح\u200cچین گوشت برام بفرثت بی\u200cذحمت', 'place_order'), ('سفارشم به مقصد اشتباه رسید', 'complaint_order'), ('دست\u200cچین گوشت برام بفرست بی\u200cزحمت', 'place_order'), ('حوبی', 'small_talk'), ('چخبر', 'small_talk'), ('سس داری', 'ask_availability'), ('بله همین بود', 'confirm_order'), ('نه گلستات 32 کوچه معتاب', 'address_change'), ('پلستان 32 کوچخ مهتاب', 'address_change'), ('گلستات 99', 'address_change'), ('تبت کن', 'confirm_order'), ('سبلم رو اضافه کن', 'add_item'), ('بلع', 'affirm'), ('نغدی میزنم', 'select_payment_method'), ('راجع به نوشیدنی ها بگو', 'ask_menu'), ('نوشیدنی ها بگو', 'ask_menu'), ('کل سبد رو حذف کن', 'remove_all_cart'), ('تاریخچه سفارشام رو بگو', 'order_status'), ('مواد مارگاریتا چیه', 'ask_ingredients'), ('منوی شما شامل چه کیک هایی میشه', 'ask_menu'), ('لوبیا دارین؟', 'ask_availability'), ('حال سلام', 'greeting'), ('هاله سلام', 'greeting'), ('حال سلام رفیق', 'greeting'), ('حاله ثلام خب', 'greeting'), ('پشت\u200cبام رو بیمه کنم', 'out_of_scope'), ('بیمه پشت\u200cبام دارین؟', 'out_of_scope'), ('پشت\u200cبام رو بیما کنی؟', 'out_of_scope'), ('قارچ سوخاری دارین؟', 'ask_availability'), ('سس قارچ', 'ask_availability'), ('نان', 'place_order', ['چه چیزی برایتان ثبت کنم؟']), ('سس\u200cهاتون چیان؟', 'ask_menu'), ('ی قهوه ترک هم بذار', 'add_item'), ('یکی دیگه اضافه کن', 'increase_quantity'), ('پرداخت در محل دارین؟', 'select_payment_method'), ('شرایط پرداخت در محل چیه؟', 'ask_restaurant_rules'), ('پیتزای سفارشم رو به مخصوص تغییر بده', 'modify_item'), ('با شماره موبایلم سفارشم رو چک می\u200cکنین؟', 'order_status'), ('پیک به این آدرس چقدر میشه', 'ask_delivery_cost'), ('ترک بر من بذار', 'unclear'), ('قهوه ترک داری', 'ask_availability'), ('چی پیشنهاد میدی؟', 'ask_menu'), ('کنسلش کن', 'cancel_order'), ('بادوم نداشته باشه', 'request_no_ingredient'), ('سفارشم کجاست؟', 'order_status'), ('پیتسا', 'place_order'), ('تخفیف دانشجویی دارین؟', 'ask_discount'), ('برای مهمونی غذا میخوام', 'catering_inquiry'), ('میشه با یه آدم صحبت کنم؟', 'human_handoff'), ('برای پنج نفر چقدر غذا لازمه؟', 'ask_menu'), ('پاستا برای یه نفر کافیه؟', 'ask_menu'), ('با شماره تلفن میشه پیگیری کرد؟', 'order_status'), ('اگه اشتباه سفارش بدم چی؟', 'ask_restaurant_rules'), ('به\u200cجای نوشابه دوتا دوغ بذار', 'modify_item'), ('پیک به این آدرس میاد؟', 'ask_delivery_zone'), ('سپاثگزارم', 'thanks'), ('ی توشابه', 'place_order'), ('نوشابه رو حذف کن', 'remove_item', ['دو تا پیتزا پپرونی و یه نوشابه میخوام'])]
    STRESS = [('گرانولا بگیرید', 'add_item'), ('ادرسم رو عوض کن', 'address_change'), ('ادرستون', 'ask_address'), ('کباب داری؟', 'ask_availability'), ('پیک به گیشا، برج نگین، زنگ دوم چند می\u200cگیره؟', 'ask_delivery_cost'), ('کی می\u200cفرستی؟', 'ask_delivery_time'), ('برای یاس، برج رز، زنگ ششم پیک دارین؟', 'ask_delivery_zone'), ('کد تخفیف دارید؟', 'ask_discount'), ('چی دارید؟', 'ask_menu'), ('پشتیبانی تلفنی دارید؟', 'ask_phone'), ('وافل چند می\u200cشه؟', 'ask_price'), ('غوانینتون چیه؟', 'ask_restaurant_rules'), ('شب بازید؟', 'ask_working_hours'), ('چت\u200cباتی؟', 'bot_identity'), ('لغو', 'cancel_order'), ('منوی کترینگ دارید؟', 'catering_inquiry'), ('پیک بی\u200cملاحظه رفتار کرد', 'complaint_courier'), ('غذا روغنی بود', 'complaint_food'), ('ثفارشم به آدرس اشتباح رفت', 'complaint_order'), ('بفرست', 'confirm_order'), ('بذار یکی بشه', 'decrease_quantity', ['دو پیتزا در سفارش شماست.']), ('نه', 'deny'), ('بای', 'goodbye'), ('سلام', 'greeting'), ('بذار سه تا بشه', 'increase_quantity', ['دو پرس چلوکباب در سبد شما ثبت شده است.']), ('تقییرش کن', 'modify_item'), ('سفارشم کو؟', 'order_status'), ('مسجد کجاست؟', 'out_of_scope'), ('وافل', 'place_order'), ('برای ثفارش ۵۳۹۳۳۴ می\u200cخوام پولمو پس بگیرم', 'refund_request'), ('یه\u200cدونح\u200cش رو بردار', 'remove_item'), ('عدسی بی نان', 'request_no_ingredient'), ('میز ۲ نفره برای دوشنبه رزرو کن', 'reservation_request'), ('نقدی', 'select_payment_method'), ('فردا چخبره؟', 'small_talk'), ('خدا پدر مادرت رو بیامرزه', 'thanks'), ('ای', 'unclear')]
    stress_fail = 0
    for _scase in STRESS:
        _stext, _sexp = (_scase[0], _scase[1])
        _sctx = _scase[2] if len(_scase) > 2 else None
        _sres = predict_message(_stext, context=_sctx)
        if _sres['intent'] != _sexp:
            stress_fail = stress_fail + 1
            print(f"❌ STRESS {_stext!r} → {_sres['intent']} (انتظار: {_sexp})")
    print(f'\n=== نتیجه\u200cی STRESS (نمونه\u200cها): {len(STRESS) - stress_fail} PASS / {stress_fail} FAIL ===')
    passed, failed, failed_cases = (0, 0, [])
    for _i, case in enumerate(REGRESSION_CASES, 1):
        _text, expected = (case[0], case[1])
        ctx = case[2] if len(case) > 2 and case[2] else None
        ent_expect = case[3] if len(case) > 3 and isinstance(case[3], dict) else {}
        res = predict_message(_text, context=ctx)
        problems = []
        if res['intent'] != expected:
            problems.append(f"intent={res['intent']} (انتظار: {expected})")
        for k, v in ent_expect.items():
            got = res['entities'].get(k)
            if got != v:
                problems.append(f'entities[{k}]={got!r} (انتظار: {v!r})')
        if problems:
            failed = failed + 1
            failed_cases.append((_text, expected, res['intent']))
            print(f"❌ [{_i}] {_text!r} → {'; '.join(problems)} | conf={res['_intent_confidence']}")
        else:
            passed = passed + 1
            print(f"✅ [{_i}] {_text!r} → {res['intent']} | conf={res['_intent_confidence']}")
    print(f'\n=== نتیجه\u200cی رگرسیون: {passed} PASS / {failed} FAIL از {len(REGRESSION_CASES)} ===')
    if failed:
        print('لیست FAILها را (همین خروجی) برای ساخت دور اصلاح بعدی بفرست.')
    else:
        print('🎉 همه\u200cی تست\u200cهای رگرسیون سبزند.')
    return


@app.cell
def _(
    DataLoader,
    EVAL_BATCH_SIZE,
    accuracy_score,
    device,
    id2bio,
    model,
    precision_recall_fscore_support,
    test_ds,
    torch,
):
    model.eval()
    test_loader = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE)
    all_intent_preds_1, all_intent_true_1 = ([], [])
    _all_secondary_preds, _all_secondary_true = ([], [])
    _all_clarif_preds, _all_clarif_true = ([], [])
    _all_ner_preds, _all_ner_true = ([], [])
    with torch.no_grad():
        for _batch in test_loader:
            _batch_on_device = {k: v.to(device) for k, v in _batch.items()}
            _outputs = model(**_batch_on_device)
            _intent_pred = _outputs['intent_logits'].argmax(-1).cpu().numpy()
            _secondary_pred = _outputs['secondary_logits'].argmax(-1).cpu().numpy()
            _clarif_pred = _outputs['clarification_logits'].argmax(-1).cpu().numpy()
            _ner_pred = _outputs['ner_logits'].argmax(-1).cpu().numpy()
            all_intent_preds_1.extend(_intent_pred.tolist())
            all_intent_true_1.extend(_batch['intent_label'].numpy().tolist())
            _all_secondary_preds.extend(_secondary_pred.tolist())
            _all_secondary_true.extend(_batch['secondary_label'].numpy().tolist())
            _all_clarif_preds.extend(_clarif_pred.tolist())
            _all_clarif_true.extend(_batch['clarification_label'].numpy().tolist())
            _ner_true_batch = _batch['ner_labels'].numpy()
            for _i in range(len(_ner_pred)):
                _seq_pred, _seq_true = ([], [])
                for _p, _t in zip(_ner_pred[_i], _ner_true_batch[_i]):
                    if _t == -100:
                        continue
                    _seq_pred.append(id2bio[_p])
                    _seq_true.append(id2bio[_t])
                _all_ner_preds.append(_seq_pred)
                _all_ner_true.append(_seq_true)
    acc_1 = accuracy_score(all_intent_true_1, all_intent_preds_1)
    _p_macro, _r_macro, f1_macro_1, _ = precision_recall_fscore_support(all_intent_true_1, all_intent_preds_1, average='macro', zero_division=0)
    _p_weighted, _r_weighted, f1_weighted_1, _ = precision_recall_fscore_support(all_intent_true_1, all_intent_preds_1, average='weighted', zero_division=0)
    print('=== Intent classification ===')
    print(f'Accuracy:      {acc_1:.4f}')
    print(f'Macro    P/R/F1: {_p_macro:.4f} / {_r_macro:.4f} / {f1_macro_1:.4f}')
    print(f'Weighted P/R/F1: {_p_weighted:.4f} / {_r_weighted:.4f} / {f1_weighted_1:.4f}')
    _sec_acc = accuracy_score(_all_secondary_true, _all_secondary_preds)
    print(f'\n=== Secondary intent ===\nAccuracy: {_sec_acc:.4f}')
    _clarif_acc = accuracy_score(_all_clarif_true, _all_clarif_preds)
    _, _, clarif_f1_1, _ = precision_recall_fscore_support(_all_clarif_true, _all_clarif_preds, average='binary', zero_division=0)
    print(f'\n=== needs_clarification ===\nAccuracy: {_clarif_acc:.4f}   F1: {clarif_f1_1:.4f}')

    def _get_entities_bio(seq):
        entities, start, etype = ([], None, None)
        for i, tag in enumerate(seq):
            if tag.startswith('B-'):
                if etype is not None:
                    entities.append((etype, start, i - 1))
                etype, start = (tag[2:], i)
            elif tag.startswith('I-') and etype == tag[2:]:
                continue
            else:
                if etype is not None:
                    entities.append((etype, start, i - 1))
                etype = None
        if etype is not None:
            entities.append((etype, start, len(seq) - 1))
        return entities

    def _entity_level_report(y_true, y_pred):
        true_ents, pred_ents = (set(), set())
        for i, (t_seq, p_seq) in enumerate(zip(y_true, y_pred)):
            true_ents = true_ents | {(i, *e) for e in _get_entities_bio(t_seq)}
            pred_ents = pred_ents | {(i, *e) for e in _get_entities_bio(p_seq)}

        def _prf(t_set, p_set):
            tp = len(t_set & p_set)
            fp = len(p_set - t_set)
            fn = len(t_set - p_set)
            pr = tp / (tp + fp) if tp + fp else 0.0
            rc = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
            return (pr, rc, f1, len(t_set))
        overall_p, overall_r, overall_f1, support = _prf(true_ents, pred_ents)
        types = sorted({e[1] for e in true_ents} | {e[1] for e in pred_ents})
        lines = [f"{'type':<20s}{'precision':>10s}{'recall':>10s}{'f1-score':>10s}{'support':>10s}"]
        for etype in types:
            t_e = {e for e in true_ents if e[1] == etype}
            p_e = {e for e in pred_ents if e[1] == etype}
            pr, rc, f1, sup = _prf(t_e, p_e)
            lines.append(f'{etype:<20s}{pr:10.4f}{rc:10.4f}{f1:10.4f}{sup:10d}')
        lines.append(f"{'micro avg':<20s}{overall_p:10.4f}{overall_r:10.4f}{overall_f1:10.4f}{support:10d}")
        return (overall_f1, '\n'.join(lines))
    print('\n=== Entity extraction (NER, entity-level) ===')
    _ner_f1, _ner_report = _entity_level_report(_all_ner_true, _all_ner_preds)
    print(f'Entity-level F1: {_ner_f1:.4f}')
    print(_ner_report)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## عیب‌یابی خطاهای رایج (Troubleshooting)

    **۱) `CUDA out of memory`**
    - `TRAIN_BATCH_SIZE` را در CELL 5 کم کنید (مثلا 16 یا 8) و `GRADIENT_ACCUMULATION_STEPS` را متناسب افزایش دهید.
    - `gradient_checkpointing=True` را در CELL 16 فعال کنید.
    - Runtime > Restart runtime و از CELL 2 دوباره اجرا کنید.

    **۲) Colab قطع شد وسط آموزش**
    - دوباره متصل شوید، CELL 2 تا CELL 16 را اجرا کنید (CELL 9 هم باید دوباره اجرا شود تا دیتاست از Hugging Face دانلود شود)،
      سپس CELL 17 را اجرا کنید — به‌صورت خودکار از آخرین checkpoint در `OUTPUT_DIR` (یا Google Drive در صورت mount بودن) ادامه می‌دهد.

    **۳) دقت پایین روی برخی intentها (طبق CELL 19/20)**
    - تعداد نمونه آن intent را در دیتاست افزایش دهید (به‌خصوص برای intentهایی که در CELL 20 با یکدیگر اشتباه می‌شوند).
    - `NUM_EPOCHS` یا `LORA_R` را در CELL 5 افزایش دهید و دوباره از CELL 15 اجرا کنید.

    **۴) خطای `اندازه دیتاست کوچک است` هنگام split**
    - اگر یک intent کمتر از ۳ نمونه دارد، برای آن intent مستقیما در دیتاست خودتان نمونه بیشتری اضافه کنید.

    **۵) entity استخراج‌شده اشتباه یا ناقص است**
    - بررسی کنید متن ورودی خارج از توزیع دیتاست train نباشد (مثلا نام غذایی که اصلا در دیتاست نبوده).
    - برای تست گروهی همه‌ی فرم‌های حساس، CELL 26 (تست رگرسیون) را اجرا کنید.
    - `CONFIDENCE_THRESHOLD` در CELL 24 را برای فعال‌تر شدن `needs_clarification` کاهش دهید (مثلا 0.4).

    **۷) `ModuleNotFoundError` برای peft/...**
    - CELL 3 را دوباره اجرا کنید؛ اگر در Colab هستید و بعد از نصب باز هم خطا گرفتید، Runtime > Restart runtime
      و دوباره از CELL 2 اجرا کنید (بعضی پکیج‌ها نیاز به ری‌استارت رانتایم دارند).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## CELL 32 — ارزیابی مقاومت STT (v46)

    مجموعه STT مستقل، خارج از train/val/test است و هیچ‌گاه در آموزش دیده نمی‌شود.
    """)
    return


@app.cell
def _(
    Dataset,
    EVAL_BATCH_SIZE,
    defaultdict,
    device,
    intent2id,
    model,
    normalize_record,
    preprocess_example,
    stt_eval_raw,
    torch,
):
    def _intent_predictions(records, batch_size=EVAL_BATCH_SIZE):
        processed = []
        for r in records:
            ex = preprocess_example(r)
            ex['_row'] = r
            processed.append(ex)
        ds = Dataset.from_list(processed)
        ds.set_format(type='torch')
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size)
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items() if k != '_row'}
                out = model(**batch)
                preds.extend(out['intent_logits'].argmax(dim=-1).cpu().tolist())
        return (preds, ds)
    stt_records = [normalize_record(r) for r in stt_eval_raw]
    stt_pred_ids, stt_ds = _intent_predictions(stt_records)
    stt_gold_ids = [intent2id[r['intent']] for r in stt_records]
    clean_records = []
    for r in stt_records:
        c = dict(r)
        c['text'] = r['clean_text']
        c['tokens'] = r['clean_text'].split()
        c['ner_tags'] = ['O'] * len(c['tokens'])
        clean_records.append(c)
    clean_pred_ids, _ = _intent_predictions(clean_records)
    corrupted_correct = sum((int(p == g) for p, g in zip(stt_pred_ids, stt_gold_ids)))
    clean_correct = sum((int(p == g) for p, g in zip(clean_pred_ids, stt_gold_ids)))
    n_stt = len(stt_records)
    destroyed = [i for i, r in enumerate(stt_records) if not r['meaning_preserved']]
    clarify_rate = sum((int(stt_pred_ids[i] == intent2id['unclear']) for i in destroyed)) / len(destroyed) if destroyed else float('nan')
    by_type = defaultdict(lambda: [0, 0])
    for _i, r in enumerate(stt_records):
        key = '+'.join(sorted(r['stt_error_types'] or []))
        by_type[key][0] = by_type[key][0] + 1
        by_type[key][1] = by_type[key][1] + int(stt_pred_ids[_i] == stt_gold_ids[_i])
    print(f'STT eval rows: {n_stt}')
    print(f'دقت روی متن خراب : {corrupted_correct / n_stt:.3f}')
    print(f'دقت روی متن سالم : {clean_correct / n_stt:.3f}')
    print(f'نرخ پرسش شفاف\u200cسازی روی نمونه\u200cهای مخدوش\u200cشده: {clarify_rate:.3f} ({len(destroyed)} نمونه)')
    print('\nبه تفکیک نوع خطا:')
    for key, (_total, correct) in sorted(by_type.items(), key=lambda kv: -kv[1][0]):
        print(f'  {key:38s} n={_total:4d} acc={correct / _total:.3f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## یادداشت روش

    - سلول‌های Auto-Fix و K-Fold (سلول‌های ۲۷ تا ۲۹) ابزار پژوهشی اختیاری‌اند و بخشی از پروتکل test مهروموم نیستند؛ برای گزارش نهایی از split صریح استفاده کنید.
    - هیچ عدد دقتی برای ParsBERT در این مخزن ادعا نشده است؛ فقط baseline تشخیصی در `reports/eval_v46.json` گزارش شده است.
    """)
    return


if __name__ == "__main__":
    app.run()
