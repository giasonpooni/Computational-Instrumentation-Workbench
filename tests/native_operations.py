"""Operations a fresh or restored workbench offers without any host binding.

Restoring a workspace never restores provider bindings, so exactly these
provider-free native references remain executable afterwards.
"""

PROVIDER_FREE_OPERATIONS = frozenset({
    "ciw.energy-accuracy.v1",
    "ciw.encoder-position.v1",
    "ciw.thermal-observer.v1",
    "ciw.project-graph.v1",
})
