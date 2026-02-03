"""
PDF Processor Module
Handles extraction of text from Arabic PDFs with proper handling of RTL text.
"""

import os
import re
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class PageContent:
    """Represents content extracted from a PDF page."""
    page_number: int
    text: str
    metadata: Dict


class PDFProcessor:
    """
    Processes PDF documents and extracts text content.
    Optimized for Arabic documents with proper text direction handling.
    """
    
    def __init__(self, pdf_path: str):
        """
        Initialize the PDF processor.
        
        Args:
            pdf_path: Path to the PDF file
        """
        self.pdf_path = pdf_path
        self.filename = os.path.basename(pdf_path)
        self.pages: List[PageContent] = []
        
    def extract_text(self) -> List[PageContent]:
        """
        Extract text from all pages of the PDF.
        
        Returns:
            List of PageContent objects containing extracted text
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF is required. Install with: pip install pymupdf")
        
        doc = fitz.open(self.pdf_path)
        self.pages = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract text with proper handling
            text = page.get_text("text")
            
            # Clean and normalize Arabic text
            text = self._clean_arabic_text(text)
            
            page_content = PageContent(
                page_number=page_num + 1,
                text=text,
                metadata={
                    "source": self.filename,
                    "page": page_num + 1,
                    "total_pages": len(doc)
                }
            )
            self.pages.append(page_content)
        
        doc.close()
        return self.pages
    
    def _clean_arabic_text(self, text: str) -> str:
        """
        Clean and normalize Arabic text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace while preserving meaningful breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Fix common OCR issues in Arabic
        text = text.replace('ى', 'ي')  # Normalize alef maksura
        text = text.replace('ﻻ', 'لا')  # Normalize lam-alef
        text = text.replace('ة', 'ة')  # Ensure proper taa marbouta
        
        # Remove any null characters
        text = text.replace('\x00', '')
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def get_full_text(self) -> str:
        """
        Get the full text content of the PDF.
        
        Returns:
            Complete text from all pages
        """
        if not self.pages:
            self.extract_text()
        
        return "\n\n".join([page.text for page in self.pages])
    
    def get_text_by_page(self, page_number: int) -> Optional[str]:
        """
        Get text from a specific page.
        
        Args:
            page_number: 1-indexed page number
            
        Returns:
            Text from the specified page, or None if not found
        """
        if not self.pages:
            self.extract_text()
        
        if 1 <= page_number <= len(self.pages):
            return self.pages[page_number - 1].text
        return None
    
    def extract_sections(self) -> List[Dict]:
        """
        Extract sections based on Arabic headings and article markers.
        
        Returns:
            List of dictionaries containing section information
        """
        if not self.pages:
            self.extract_text()
        
        full_text = self.get_full_text()
        
        # Pattern to match Arabic article markers like "مادة (1):" or "مادة :)1("
        article_pattern = r'مادة\s*[:\(]?\s*(\d+)\s*[:\)]?'
        
        sections = []
        matches = list(re.finditer(article_pattern, full_text))
        
        for i, match in enumerate(matches):
            start_pos = match.start()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
            
            section_text = full_text[start_pos:end_pos].strip()
            article_number = match.group(1)
            
            # Extract the title (first line after article marker)
            lines = section_text.split('\n')
            title = lines[0] if lines else f"Article {article_number}"
            
            sections.append({
                "article_number": int(article_number),
                "title": title,
                "content": section_text,
                "metadata": {
                    "source": self.filename,
                    "article": article_number
                }
            })
        
        return sections


def extract_pdf_to_json(pdf_path: str, output_path: str) -> None:
    """
    Extract PDF content and save to JSON file.
    
    Args:
        pdf_path: Path to the PDF file
        output_path: Path for the output JSON file
    """
    import json
    
    processor = PDFProcessor(pdf_path)
    pages = processor.extract_text()
    
    data = {
        "source": processor.filename,
        "total_pages": len(pages),
        "pages": [
            {
                "page_number": page.page_number,
                "text": page.text,
                "metadata": page.metadata
            }
            for page in pages
        ],
        "sections": processor.extract_sections()
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Extracted {len(pages)} pages to {output_path}")
