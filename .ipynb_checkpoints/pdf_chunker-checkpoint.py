#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import re
from pathlib import Path
import requests
import shutil
from typing import List, Dict, Tuple
from collections import defaultdict

# Assumption: You have installed the necessary libraries
# pip install requests langchain-community langchain-core pymupdf
try:
    from langchain_core.documents import Document
    from langchain_pymupdf4llm import PyMuPDF4LLMLoader
except ImportError:
    raise ImportError(
        "LangChain libraries not found. Please install with: "
        "`pip install langchain-community langchain-core pymupdf`"
    )

class PDFChunkerForQdrant:
    """
    Processes one or more PDFs according to a specific 5-step algorithm,
    preparing documents for storage in a vector database like Qdrant.
    """

    def __init__(self, max_char_limit: int):
        if not isinstance(max_char_limit, int) or max_char_limit <= 0:
            raise ValueError("max_char_limit must be a positive integer.")
        self.max_char_limit = max_char_limit
        self.download_dir = Path("./temp_pdf_downloads")
        self.download_dir.mkdir(exist_ok=True)

    def __del__(self):
        if self.download_dir.exists():
            shutil.rmtree(self.download_dir)
            print("\nCleaned up temporary download directory.")

    def process_pdfs(self, pdf_sources: List[str]) -> List[Document]:
        """Processes a list of PDFs from URLs or local paths."""
        all_documents = []
        print(f"--- Starting batch processing for {len(pdf_sources)} source(s) ---")
        for source in pdf_sources:
            try:
                documents_from_one_pdf = self._process_single_pdf(source)
                if documents_from_one_pdf:
                    all_documents.extend(documents_from_one_pdf)
            except Exception as e:
                print(f"--- ❌ Critical error processing '{source}': {e}. Skipping. ---")

        print(f"\n--- ✅ Batch processing complete. Generated a total of {len(all_documents)} documents. ---")
        return all_documents

    def _process_single_pdf(self, pdf_source: str) -> List[Document]:
        """Orchestrates the 5-step processing pipeline for a single PDF."""
        try:
            print(f"\n--- Starting processing for: {pdf_source} ---")

            file_name, pages = self._load_and_convert_pdf(pdf_source)
            if not pages:
                return []

            initial_chunks_data = self._create_initial_chunks(pages)
            print(f"Steps 2 & 3: Divided into {len(initial_chunks_data)} initial chunks.")

            consolidated_data = self._consolidate_chunks(initial_chunks_data)
            print(f"Step 4: Consolidated into {len(consolidated_data)} final chunks.")
            #for chunk in consolidated_data:
                #print("\nchunky--------------------\n", chunk)

            final_documents = self._create_langchain_documents(consolidated_data, file_name)

            print(f"--- ✅ Successfully processed '{file_name}' into {len(final_documents)} documents. ---")
            return final_documents

        except (IOError, FileNotFoundError, requests.RequestException, ValueError) as e:
            print(f"--- ❌ Error processing '{pdf_source}': {e}. Skipping this file. ---")
            return []

    def _load_and_convert_pdf(self, pdf_source: str) -> Tuple[str, List[Document]]:
        """Step 1: Downloads/finds PDF and converts to markdown pages."""
        if pdf_source.startswith("http"):
            response = requests.get(pdf_source, timeout=30)
            response.raise_for_status()
            base_name = os.path.basename(pdf_source.split("?")[0])
            if not base_name.lower().endswith('.pdf'):
                base_name = "download.pdf"

            pdf_path = self.download_dir / f"{Path(base_name).stem}_{os.urandom(4).hex()}.pdf"
            pdf_path.write_bytes(response.content)
            source_path_for_loader = str(pdf_path)
            file_name = base_name
            print(f"Downloaded '{file_name}' to '{pdf_path}'")
        else:
            pdf_path = Path(pdf_source)
            if not pdf_path.exists():
                 raise FileNotFoundError(f"Local PDF file not found at: {pdf_source}")
            file_name = pdf_path.name
            source_path_for_loader = str(pdf_path)

        print(f"Loading and converting '{file_name}' to markdown...")
        loader = PyMuPDF4LLMLoader(source_path_for_loader)
        # Add the file_name to each page's metadata right away
        loaded_pages = loader.load()
        for page in loaded_pages:
            page.metadata['file_name'] = file_name
        return file_name, loaded_pages

    def _create_initial_chunks(self, pages: List[Document]) -> List[Dict]:
        """Steps 2 & 3: Identify sections and chunk them by page, tracking page numbers."""
        file_name = pages[0].metadata['file_name']
        prefix = os.path.splitext(file_name)[0][:2]
        header_pattern = re.compile(rf"^\s*(\*\*{re.escape(prefix)}[^\*]+\*\*)\s*$", re.MULTILINE)

        sections = []
        current_section_pages = []

        for page in pages:
            page_num = page.metadata.get('page', 0)
            content = page.page_content
            headers_on_page = list(header_pattern.finditer(content))

            if not headers_on_page:
                current_section_pages.append({'num': page_num, 'content': content})
            else:
                last_pos = 0
                for match in headers_on_page:
                    pre_header_content = content[last_pos:match.start()]
                    if pre_header_content.strip():
                        current_section_pages.append({'num': page_num, 'content': pre_header_content})

                    if current_section_pages:
                        sections.append(current_section_pages)

                    current_section_pages = [{'num': page_num, 'content': match.group(0)}]
                    last_pos = match.end()

                remaining_content = content[last_pos:]
                if remaining_content.strip():
                    # This content belongs to the new section started by the last header
                    current_section_pages.append({'num': page_num, 'content': remaining_content})

        if current_section_pages:
            sections.append(current_section_pages)

        initial_chunks = []
        for section_pages in sections:
            section_content = "".join([p['content'] for p in section_pages])

            if len(section_content) <= self.max_char_limit:
                page_nums = [p['num'] for p in section_pages]
                initial_chunks.append({'content': section_content, 'pages': page_nums})
            else:
                current_chunk_content = ""
                current_chunk_pages = []

                for page_data in section_pages:
                    page_len = len(page_data['content'])
                    if current_chunk_content and len(current_chunk_content) + page_len > self.max_char_limit:
                        initial_chunks.append({'content': current_chunk_content, 'pages': current_chunk_pages})
                        current_chunk_content = page_data['content']
                        current_chunk_pages = [page_data['num']]
                    else:
                        current_chunk_content += page_data['content']
                        current_chunk_pages.append(page_data['num'])

                if current_chunk_content:
                    initial_chunks.append({'content': current_chunk_content, 'pages': current_chunk_pages})

        return initial_chunks

    def _consolidate_chunks(self, chunks_data: List[Dict]) -> List[Dict]:
        """Step 4: Combines smaller chunks, merging their content and page lists."""
        if not chunks_data:
            return []

        consolidated = []
        current_chunk = chunks_data[0].copy()
        separator = "\n\n---\n\n"

        for next_chunk in chunks_data[1:]:
            if len(current_chunk['content']) + len(separator) + len(next_chunk['content']) <= self.max_char_limit:
                current_chunk['content'] += separator + next_chunk['content']
                current_chunk['pages'].extend(next_chunk['pages'])
            else:
                consolidated.append(current_chunk)
                current_chunk = next_chunk.copy()

        consolidated.append(current_chunk)
        return consolidated

    def _format_page_numbers(self, pages: List[int]) -> str:
        """Converts a list of page numbers like [0, 1, 2, 4] to '1-3, 5'."""
        if not pages:
            return ""

        # Correct for 0-based index and get unique, sorted pages
        page_nums = sorted(list(set(p + 1 for p in pages)))

        ranges = []
        start = end = page_nums[0]

        for i in range(1, len(page_nums)):
            if page_nums[i] == end + 1:
                end = page_nums[i]
            else:
                ranges.append(str(start) if start == end else f"{start}-{end}")
                start = end = page_nums[i]

        ranges.append(str(start) if start == end else f"{start}-{end}")
        return ", ".join(ranges)

    def _create_langchain_documents(self, consolidated_data: List[Dict], file_name: str) -> List[Document]:
        """Step 5: Creates final Langchain Document objects with detailed metadata."""
        final_documents = []
        for chunk_data in consolidated_data:
            formatted_pages = self._format_page_numbers(chunk_data['pages'])

            # Create the full content with the required header
            full_content = (
                f"File: {file_name}\n"
                f"Pages: {formatted_pages}\n\n"
                f"{chunk_data['content']}"
            )

            doc = Document(
                page_content=full_content,
                metadata={'file_name': file_name}
            )
            final_documents.append(doc)
        return final_documents

