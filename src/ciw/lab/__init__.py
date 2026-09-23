"""Computational-experimentalist queue for the workbench.

Each queued task follows one loop: hypothesis, mathematical prediction,
synthetic or physical protocol, implementation, execution, independent
comparison, uncertainty/provenance report, then a regression test or a
deferred research question. Every finding carries one evidence label from
:mod:`ciw.lab.evidence`; physical results are never fabricated, and tasks that
need unavailable hardware or providers are reported as blocked or deferred.
"""
