"""Explicit reference-counted ownership for immutable stationary preparation."""
from dataclasses import dataclass
from .surface_chart import build_chart
from .contact_field import build_contact_field


@dataclass
class PreparedSource:
    source: object
    chart: object
    contact: object
    fingerprint: str
    references: int = 1

    def retain(self):
        if self.references == 0:
            raise RuntimeError('Prepared source is closed')
        self.references += 1
        return self

    def release(self):
        if self.references == 0: return
        self.references -= 1
        if self.references == 0:
            self.chart.close(); self.contact.close(); self.source.close()


def prepare_source(source, fingerprint: str, field_spacing: float, contact_spacing: float) -> PreparedSource:
    chart = contact = None
    try:
        chart = build_chart(source,field_spacing)
        contact = build_contact_field(source,contact_spacing)
        return PreparedSource(source,chart,contact,fingerprint)
    except Exception:
        if contact is not None: contact.close()
        if chart is not None: chart.close()
        source.close()
        raise
