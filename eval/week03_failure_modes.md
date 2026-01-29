1) q05 — “When should you not move the Acorn Archimedes computer?”  
   - Gold: `Acorn Archimedes Guide_00012` | Retrieved: earlier intro/setup pages (e.g. `_00001.._00008`)  
   - Likely cause: wording mismatch (“do not move” vs “transport/relocate/handle”), plus chunk boundary (warning text sits in a different section than “moving” keywords).

2) q07 — “What operating systems are supported by the Compaq AlphaServer DS10 systems?”  
   - Gold: `...QuickSpecs..._00002` | Retrieved: other QuickSpecs pages (`_00005`, `_00028`, `_00029`)  
   - Likely cause: answer is in a spec table / bullet list where OS names are short tokens; TF‑IDF/BM25 can drift to other pages mentioning “systems/support” more often.

3) q10 — “What is the nominal operating voltage of the Compaq AlphaServer DS20?” *(TF‑IDF failure)*  
   - Gold: `DS20 QuickSpecs (2000)_00024` | Retrieved: DS10 `_00029`, DS20 `_00023/_00015`  
   - Likely cause: tables/specs (voltage appears as a small numeric field), and document confusion (DS10 vs DS20 share many overlapping terms like “operating/voltage/power”).

4) q11 — “What voltage ranges are supported by the Compaq AlphaServer GS80 power supplies?”  
   - Gold: `GS80 Installation Guide (2000)_00012` | Retrieved: DS10 `_00029`, DS20 QuickSpecs, GS80 `_00004`  
   - Likely cause: missing table signal (ranges like “100–240 VAC” are weakly represented in bag‑of‑words), plus chunking (range may be split across lines/columns after PDF extraction).

5) q12 — “What environmental conditions are specified for operating the GS80?”  
   - Gold: `GS80 Installation Guide (2000)_00014` | Retrieved: GS80 `_00004` + unrelated docs  
   - Likely cause: spec table (temperature/humidity/altitude), and PDF noise (headers/footers can dominate term frequency; “environmental conditions” may not appear verbatim).

6) q14 — “What is the operating voltage range for the DS20?” *(TF‑IDF failure)*  
   - Gold: `DS20 QuickSpecs (2000)_00024` | Retrieved: OpenVMS “System Messages” pages + DS10 `_00029`  
   - Likely cause: keyword collision on generic words (“operating”, “range”), and insufficient normalization (query not biased to “DS20 QuickSpecs” vs other corpora).

7) q19 — “What AC voltage range is supported by Tru64 UNIX AlphaServer systems?”  
   - Gold: `Tru64 UNIX ...AdvFS Administration..._00018` | Retrieved: DS10/DS20 QuickSpecs + Tru64 index page  
   - Likely cause: wrong document family (hardware electrical spec is likely in a hardware quickspec, not an AdvFS admin guide); gold chunk may contain mention, but retrieval gravitates to “AC voltage” occurrences in other hardware docs.

## 3 Improvement Proposals

1) Spec/table-aware text prep  
   - Better PDF cleanup for tables (preserve row/column text, normalize “100–240 VAC” tokens, remove repeated headers/footers/TOC junk).

2) Chunking tuned for manuals  
   - Smaller chunks + overlap around headings; keep tables + their captions together; avoid splitting “Voltage range:” across chunks.

3) Query normalization + lightweight expansion  
   - Normalize synonyms (“move”→“transport/relocate/handling”), add model bias tokens (“DS20 QuickSpecs”) and numeric pattern boosting (VAC/VDC ranges), or use simple hybrid (BM25 + TF‑IDF) rerank.

