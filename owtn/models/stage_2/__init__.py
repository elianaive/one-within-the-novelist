"""Stage 2 data models. Import from submodules:

    from owtn.models.stage_2.dag import DAG, Node, Edge, MotifMention
    from owtn.models.stage_2.config import Stage2Config
    from owtn.models.stage_2.handoff import Stage1Winner, Stage2Output
    from owtn.models.stage_2.mcts_node import MCTSNode
    from owtn.models.stage_2.pacing import get_preset

Empty `__init__.py` (matches Stage 1 convention) — re-exports here cause
`python -m owtn.models.stage_2.<module>` to emit a RuntimeWarning about
double-import. Submodule paths are explicit and unambiguous.
"""
