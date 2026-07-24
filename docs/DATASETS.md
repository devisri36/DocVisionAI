# Dataset Reference Guide

DocVision AI supports automated loaders and pipelines for four standard document datasets:

---

## 1. FUNSD (Form Understanding in Noisy Scanned Documents)
- **Task**: Key-Value relation extraction, entity classification.
- **Data Structure**:
  - `annotations/`: JSON files containing bounding boxes (`box`), text strings, labels (`question`, `answer`, `header`), and link pairings.
  - `images/`: Scanned noisy documents.
- **Florence Prompt Target**: `<OCR_WITH_REGION>` Task.
  - target representation: `word1 <loc_y1><loc_x1><loc_y2><loc_x2> (label) | ...`

---

## 2. CORD (Consolidated Receipt Dataset)
- **Task**: Post-OCR Parsing, hierarchical menu extraction.
- **Data Structure**:
  - `annotations/`: Hierarchical nested JSON receipt items and prices.
  - `images/`: Thermal receipt scans.
- **Florence Prompt Target**: `<OCR_WITH_REGION>` Task.

---

## 3. SROIE (Scanned Receipts OCR and Information Extraction)
- **Task**: Named Entity Recognition (NER) on receipt scans (Company, Date, Address, Total).
- **Data Structure**:
  - `annotations/`: JSON files mapping OCR tokens and the correct values for target keys.
  - `images/`: Scanned cash register receipts.
- **Florence Prompt Target**: `<KIE>` Task.
  - target representation: `company: X | date: Y | address: Z | total: W`

---

## 4. DocVQA (Document Visual Question Answering)
- **Task**: Visual VQA querying.
- **Data Structure**:
  - `annotations/`: Question strings, list of correct answers, OCR words and bboxes.
  - `images/`: Scanned documents.
- **Florence Prompt Target**: `<DocVQA>` Task.
  - target representation: `Question: X -> Answer: Y`
