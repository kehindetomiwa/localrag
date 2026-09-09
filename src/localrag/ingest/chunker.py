"""Structure-aware chunking via LangChain text splitters.

Splits documents along markdown header boundaries first (`MarkdownHeaderTextSplitter`)
so chunks map onto logical sections (e.g. "Overview", "Setup", "API"), then
runs a token-window split with overlap (`TokenTextSplitter`) over any section
that's still too large. Because sibling documents in the same doc_group share
a template, this keeps same-named sections comparable across documents, while
the doc_name/doc_group tags added during ingestion keep them distinguishable
at retrieval time.
"""
from __future__ import annotations

import tiktoken #Used to calculate token cost
from langchain_core.documents import Document
#Markdown splitter is used to split markdown files cleaner while token text splits the text into smaller overlapping chunks
from langchain_text_splitters import MarkdownHeaderTextSplitter, TokenTextSplitter

from ..config import settings

# This specifies the range we want to split based off the ammount of headers #--> output is #, "h1"
_HEADERS_TO_SPLIT_ON = [("#" * n, f"h{n}") for n in range(1, 7)]
#Initializing our encoder.. the cl100k_base is the model used by openai
_encoding = tiktoken.get_encoding("cl100k_base")


#helper function to extract the title of the section
def _section_title(section: Document) -> str:
    #Looping through each titles from down to up to get the most specific title so it starts from h6 up to h1
    for n in range(6, 0, -1):
        title = section.metadata.get(f"h{n}")
        if title:
            return title
    return "Document"


#This is the main function that does the splitting
def chunk_document(
    content: str, #What it contains
    max_tokens: int | None = None, # How many tokens we want to split the document into
    overlap_tokens: int | None = None,# How many tokens we want to overlap between each chunk
) -> list[Document]:
    #If we provide a max tokens we use that otherwise we use the default same as the overlap
    max_tokens = max_tokens or settings.chunk_max_tokens
    overlap_tokens = overlap_tokens or settings.chunk_overlap_tokens

    #Creating the splitter for the markdown files when strip_headers is false so we can grab the \n
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS_TO_SPLIT_ON, strip_headers=False
    )
    token_splitter = TokenTextSplitter(
        encoding_name="cl100k_base", chunk_size=max_tokens, chunk_overlap=overlap_tokens
    )
    #Chunks is the variable name and list[documents] is just descriin what type of data we want to collect in the list 
    chunks: list[Document] = []
    #Looping through the documents of the splitted content
    for section in header_splitter.split_text(content):
        #Extracting the title
        title = _section_title(section)

        #Takes the content from a title then split it into smaller chunks
        for sub_text in token_splitter.split_text(section.page_content):
            if not sub_text.strip():
                continue
            chunks.append(
                Document(
                    page_content=sub_text,
                    metadata={
                        "section_title": title,
                        "chunk_index": len(chunks),
                        "token_count": len(_encoding.encode(sub_text)),
                    },
                )
            )
    return chunks


#Sample Output
# Document(
#         page_content="## Usage\nImport the main class and call run().",
#         metadata={
#             "section_title": "Usage",
#             "chunk_index": 1,
#             "token_count": 12
#         }
