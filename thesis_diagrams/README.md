# Thesis Diagrams

Mermaid source files for updated VCAI thesis diagrams.
Paste each ```mermaid ... ``` block at https://mermaid.live to generate PNG/SVG images.

| File | Replaces / Adds | Status |
|------|----------------|--------|
| 01_system_overview.md | Replaces `images/SystemOver.png` | RAG moved to eval-only; Adaptive Learning Loop added |
| 02_conversation_sequence.md | Replaces `images/sequence_diagram.png` | RAGAgent removed from live conversation |
| 03_evaluation_pipeline.md | New / supplements `images/RAGUML.png` | Full two-pass eval + hybrid RAG flowchart |
| 04_adaptive_learning_loop.md | New diagram for new section in Ch4 | Shows skill tracking → recommendation feedback loop |
| 05_erd.md | Replaces `images/ERD.png` | Adds skill score columns and training_focus to sessions |

## Screenshots needed (take from running app)
- Dashboard showing skill progress widget (3 weakest skills + trend arrows)
- Progress page showing skill trend charts
- Session Setup page showing recommended persona card
- Evaluation report page (current look)
