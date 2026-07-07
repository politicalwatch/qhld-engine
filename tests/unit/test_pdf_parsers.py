from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from qhld_engine.extractors.spain.initiative_extractors.utils import pdf_parsers
from qhld_engine.extractors.spain.initiative_extractors.utils.pdf_parsers import (
    PDFParser,
)

pytestmark = pytest.mark.unit

_STREAM = (
    b"BT /F1 18 Tf 72 720 Td (Proyecto de Ley 12/2024) Tj "
    b"0 -28 Td (Comision de Justicia) Tj ET"
)
SAMPLE_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
    b"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
    b"4 0 obj<</Length " + str(len(_STREAM)).encode() + b">>stream\n"
    + _STREAM + b"\nendstream endobj\n"
    b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
    b"trailer<</Root 1 0 R/Size 6>>\n%%EOF"
)


def test_extract_returns_text_lines_from_real_pdf():
    lines = PDFParser(SimpleNamespace(name=BytesIO(SAMPLE_PDF))).extract()

    assert "Proyecto de Ley 12/2024" in lines
    assert "Comision de Justicia" in lines


def test_extract_postprocesses_form_feed_tabs_and_newlines():
    raw = "  Titulo\fSegunda\tpagina\nTercera  "

    with patch.object(pdf_parsers, "extract_text", return_value=raw):
        result = PDFParser(SimpleNamespace(name=None)).extract()

    assert result == ["Titulo Segundapagina", "Tercera"]


def test_extract_returns_empty_list_when_extraction_fails():
    assert PDFParser(SimpleNamespace(name=BytesIO(b"not a pdf"))).extract() == []
