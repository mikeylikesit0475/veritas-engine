"""OTel telemetry setup — Sprint 4.3.

ADK already instruments the graph with OpenTelemetry spans. This module configures
the OTel SDK to export those spans to stdout so they're visible in the demo.

For the README, a screenshot of a trace (spans covering the whole graph) is the
deliverable. The trace shows: intake → triage → decompose → fan_out (parallel
spans for skeptic + steelman) → arbiter → confidence_route → report.

In production, swap ``ConsoleSpanExporter`` for an OTLP exporter to Jaeger/Tempo.
For the PoC, stdout is sufficient — an interviewer can see the spans in the
terminal.
"""

from __future__ import annotations

import logging

__all__ = ["setup_telemetry"]

logger = logging.getLogger(__name__)


def setup_telemetry() -> None:
    """Configure OTel to export spans to stdout.

    Called once at server startup. ADK's built-in instrumentation handles the
    rest — every node, tool call, and model call becomes a span.
    """
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )

    provider = TracerProvider(resource=None)
    provider.add_span_processor(
        BatchSpanProcessor(ConsoleSpanExporter())
    )
    trace.set_tracer_provider(provider)
    logger.info("OTel telemetry configured — spans exported to stdout.")
