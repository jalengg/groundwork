# Tax Intake System — Design Document
**Date:** 2026-02-28
**Status:** Approved
**Audience:** Developer (Jalen)

---

## Overview

A self-hosted, privacy-first AI-powered intake and processing system for a small Chinese-speaking tax accounting practice. All client data stays on a dedicated local mini PC — no cloud APIs ever touch client documents.

The system has three layers:
1. **Reception Center** — a self-hosted email inbox where all client documents are forwarded
2. **Triage Agent** — a background AI service that processes incoming emails (OCR, STT, extraction)
3. **Review & Assistant App** — a PyQt6 desktop application for review, correction, export, and per-client AI assistance

---

## Architecture

```
[WeChat / Email / iMessage / Other]
         |
         | (parents manually forward)
         v
┌─────────────────────────────────┐
│   Reception Center              │  Mailcow (Docker, self-hosted SMTP/IMAP)
│   intake@local or real domain   │  All content arrives as emails/attachments
└─────────────┬───────────────────┘
              |
              v
┌─────────────────────────────────┐
│   Triage Agent (background)     │  Python service polling IMAP every 60s
│   OCR · STT · AI Extraction     │  Qwen2.5-VL-7B + Whisper + PaddleOCR
└─────────────┬───────────────────┘
              |
              v
┌─────────────────────────────────┐
│   Review & Assistant App        │  PyQt6 desktop application
│   Review · Correct · Export     │  Intake review + per-client AI assistant
│   Per-Client AI Assistant       │  DeepSeek-R1-8B (reasoning, on demand)
└─────────────────────────────────┘
```

---

## Section 1: Reception Center

**Technology:** Mailcow (Docker-based self-hosted email server)

Mailcow provides a full IMAP/SMTP server with a web admin UI. It runs on the mini PC and is accessible on the local network. The intake address (`intake@local` or a real domain) receives all forwarded client content.

### Forwarding Workflow

| Source | How parents forward |
|---|---|
| WeChat images/files | WeChat Desktop → right-click → Forward → email to intake address |
| WeChat voice messages | WeChat Desktop saves locally → attach to email → forward |
| WeChat text descriptions | Copy-paste into email body → send to intake address |
| Client emails | Clients email the intake address directly (zero manual step) |
| iMessage/SMS | Screenshot → attach to email → forward |

### Why Email as the Hub
- Universal protocol — works from any device, any app
- Supports attachments of any type (images, PDFs, audio, text)
- Self-hostable with no external dependencies
- Natural audit trail (every intake item is a timestamped email)

---

## Section 2: Triage Agent

**Technology:** Python background service (runs as a system service on the mini PC)

Polls the IMAP inbox every 60 seconds. For each new unseen email:

### Processing Pipeline

```
New email detected
      |
      ├── Email body (text) ──────────────────→ Qwen: extract structured info
      |
      ├── Image attachments (.jpg, .png, etc.)
      │         ├── PaddleOCR: extract raw Chinese/English text
      │         └── Qwen2.5-VL: understand document type + fields
      |
      ├── PDF attachments
      │         ├── pdfplumber: extract embedded text
      │         └── Qwen: classify + extract fields
      |
      └── Audio attachments (.m4a, .mp3, .silk, .amr)
                ├── ffmpeg: convert to WAV
                ├── Whisper (large-v3): Chinese STT → transcript
                └── Qwen: summarize + extract intent/amounts
```

### Output: Structured Intake Record

Each processed email produces one intake record stored in SQLite:

```
intake_record:
  id
  received_at
  email_subject
  email_from
  raw_email_body
  attachments[]          -- file paths to stored originals
  document_type          -- W-2, 1099, receipt, expense, voice note, other
  client_name_guess      -- extracted from content or email sender
  client_id              -- null until matched/assigned in review app
  tax_year_guess
  extracted_fields       -- JSON: amounts, dates, employer names, etc.
  raw_ocr_text
  whisper_transcript     -- if audio
  confidence_score       -- 0.0–1.0
  status                 -- pending | reviewed | exported | flagged
  notes                  -- reviewer notes
```

### Privacy Guarantee
- Originals stored in `/data/intake/YYYY/MM/DD/<uuid>/` on local disk
- No content sent to any external API
- SQLite database stored locally, not synced to cloud

---

## Section 3: Review Desktop App

**Technology:** PyQt6 (Python), dense maximalist UI

Designed for users comfortable with 2000s-era software: all controls visible, information-dense, consistent layout, no hidden menus.

### Views

#### 3.1 Intake Inbox
- Table of all pending intake records
- Columns: date received, client (guessed), document type, status, confidence score, tax year
- Color-coded by status: new (white), reviewed (green), flagged (yellow), exported (blue)
- Sortable, filterable by status/client/date/type
- Double-click opens Review Pane

#### 3.2 Review Pane
- Left panel: original attachment viewer (image zoom, PDF viewer, audio player with waveform)
- Right panel: editable form fields populated from extraction
  - Client name (with autocomplete from client list)
  - Document type (dropdown)
  - Tax year
  - Key extracted fields (amounts, dates, payer/employer, etc.)
  - Raw OCR text (expandable)
  - Reviewer notes
- Action buttons: Approve, Flag, Reject, Assign to Client

#### 3.3 Client Ledger
- Per-client view of all approved records across tax years
- Grouped by year → by document type
- Running totals calculated per category
- Export buttons per year or per client

#### 3.4 Export Panel
- Select records by client, year, or status
- Export formats:
  - Excel `.xlsx` — structured spreadsheet with one tab per document type
  - QuickBooks Desktop — IIF format
  - QuickBooks Online — CSV in QBO-expected format
- Export history log

#### 3.5 Client Management
- Client list with contact info, preferred language, notes
- Link/merge intake records to clients
- Per-client document history

---

## Section 4: Per-Client AI Assistant Mode

A dedicated screen accessible from the Client Ledger. Modeled after Claude Code's UI: streaming chat, tool outputs rendered inline, full conversation history per client.

### Context Siloing

Each client has a strictly isolated context:
- Separate SQLite partition (or filtered views with client_id enforcement)
- Separate document folder on disk
- Separate conversation history
- When assistant opens for Client X, system prompt contains **only** Client X's data
- No cross-client data ever appears in the same context window
- Enforced at application layer (not just by prompt)

### What the Assistant Can Do

| Capability | Description |
|---|---|
| Answer math questions | Reads client's extracted fields, reasons step-by-step (estimated quarterly payments, deduction totals, etc.) |
| Merge/combine Excel sheets | Reads client's Excel files, generates merged output, shows inline in chat |
| Prepare QB bulk upload | Generates QuickBooks IIF or CSV from all approved records for a client/year |
| Summarize a tax year | Plain-language summary of all approved documents for a year |
| Flag anomalies | Detects mismatches, unusual amounts, missing documents |
| Answer freeform questions | "Did they submit their Schedule C receipts?" / "What's missing for 2024?" |

### UI
- Left sidebar: client list + conversation history (date-stamped sessions per client)
- Main area: streaming chat interface, tool call outputs rendered inline (tables, file links, step-by-step calculations)
- Language: responds in Chinese by default, toggle to English per session
- Conversation persists — next session continues from last

### Model
- **DeepSeek-R1-8B** via Ollama (loaded on demand, swapped out when assistant is closed)
- Strong reasoning, math, structured output generation
- Runs on the same mini PC, no VRAM conflict with Qwen when assistant is active

---

## Section 5: Hardware & Stack

### Recommended Mini PC
| Spec | Recommendation |
|---|---|
| CPU | AMD Ryzen 9 or Intel Core Ultra |
| RAM | 32GB DDR5 |
| GPU | NVIDIA RTX 4060 (8GB VRAM) — runs one model at a time comfortably |
| Storage | 1TB NVMe SSD (OS + models + client data) |
| Form factor | Beelink SER8, Minisforum UM890 Pro, or similar |
| Upgrade path | RTX 4070 (12GB VRAM) if both models need to run simultaneously |

### Full Stack

| Component | Technology | Purpose |
|---|---|---|
| Email server | Mailcow (Docker) | Self-hosted SMTP/IMAP reception center |
| Triage agent | Python + imaplib/email | Inbox polling and routing |
| OCR | PaddleOCR | Chinese-optimized document OCR |
| Vision/language AI | Qwen2.5-VL-7B via Ollama | Document understanding, extraction |
| Speech-to-text | Whisper large-v3 (local) | Chinese voice message transcription |
| Audio conversion | ffmpeg | Convert WeChat audio formats to WAV |
| Reasoning AI | DeepSeek-R1-8B via Ollama | Per-client assistant, math, QB prep |
| Desktop app | PyQt6 + Python | Main UI |
| Database | SQLite | Local intake records, client data, conversations |
| Excel export | openpyxl | `.xlsx` generation |
| QuickBooks export | Custom IIF/CSV generator | QB Desktop + Online import |
| Model serving | Ollama | Local model management and inference |
| Containerization | Docker Compose | Mailcow + supporting services |

---

## Privacy Model

- **Client documents never leave the mini PC** at any point in the pipeline
- **No cloud AI APIs** used for any client data (Ollama serves all models locally)
- **Claude Code / proprietary AI** used only for development tooling — never sees client data
- **Mailcow** runs locally — if the mini PC is not exposed to the internet, the email server is LAN-only
- **SQLite** database is not synced to any cloud service
- **Backups** should be encrypted local backups (e.g. to an external drive), not cloud

---

## Out of Scope (v1)

- Direct WeChat API integration (no public API exists)
- Direct QuickBooks API integration (can be added in v2)
- Mobile app for parents
- Multi-user / multi-accountant support
- Client-facing portal
- Automated tax form generation

---

## Approval

Design reviewed and approved on 2026-02-28.
