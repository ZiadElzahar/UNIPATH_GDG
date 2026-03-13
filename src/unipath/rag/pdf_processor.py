"""PDF Processor Module."""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class PageContent:
    page_number: int
    text: str
    metadata: Dict


class PDFProcessor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.filename = os.path.basename(pdf_path)
        self.pages: List[PageContent] = []

    def extract_text(self) -> List[PageContent]:
        import fitz

        doc = fitz.open(self.pdf_path)
        self.pages = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = self._clean_arabic_text(page.get_text("text"))
            self.pages.append(
                PageContent(
                    page_number=page_num + 1,
                    text=text,
                    metadata={"source": self.filename, "page": page_num + 1, "total_pages": len(doc)},
                )
            )
        doc.close()
        return self.pages

    def _clean_arabic_text(self, text: str) -> str:
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        text = text.replace("ى", "ي").replace("ﻻ", "لا")
        text = text.replace("\x00", "")
        return text.strip()

    def get_full_text(self) -> str:
        if not self.pages:
            self.extract_text()
        return "\n\n".join([page.text for page in self.pages])

    def get_text_by_page(self, page_number: int) -> Optional[str]:
        if not self.pages:
            self.extract_text()
        if 1 <= page_number <= len(self.pages):
            return self.pages[page_number - 1].text
        return None

    def extract_sections(self) -> List[Dict]:
        if not self.pages:
            self.extract_text()

        full_text = self.get_full_text()
        article_pattern = r"مادة\s*[:\(]?\s*(\d+)\s*[:\)]?"
        sections = []
        matches = list(re.finditer(article_pattern, full_text))

        for i, match in enumerate(matches):
            start_pos = match.start()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
            section_text = full_text[start_pos:end_pos].strip()
            article_number = match.group(1)
            lines = section_text.split("\n")
            title = lines[0] if lines else f"Article {article_number}"
            sections.append(
                {
                    "article_number": int(article_number),
                    "title": title,
                    "content": section_text,
                    "metadata": {"source": self.filename, "article": article_number},
                }
            )
        return sections


def extract_pdf_to_json(pdf_path: str, output_path: str) -> None:
    import json

    processor = PDFProcessor(pdf_path)
    pages = processor.extract_text()

    data = {
        "source": processor.filename,
        "total_pages": len(pages),
        "pages": [{"page_number": page.page_number, "text": page.text, "metadata": page.metadata} for page in pages],
        "sections": processor.extract_sections(),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
