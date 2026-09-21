from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def find_files(data_dir: str | Path) -> list[Path]:
    return sorted(Path(data_dir).rglob("*.md"))

def chunk_file(file_dir: str | Path) -> list[Document]:
        file_text = clean_markdown(Path(file_dir).read_text(encoding="utf-8"))
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[('#', 'h1'), ('##', 'h2'), ('###', 'h3')])
        md_header_split = markdown_splitter.split_text(file_text)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(md_header_split)
        return splits

def clean_markdown(text: str) -> str:
    out, in_fence = [], False

    for line in text.splitlines():
        stripped = line.strip()

        if(stripped.startswith("```")):
            in_fence = not in_fence
            out.append(line)
            continue
        
        if not in_fence:
            if(stripped.startswith("{*")) and stripped.endswith("*}"):
                continue
            if stripped.startswith("///"):
                continue
        
        out.append(line)
    return "\n".join(out)


def split_anchor(heading: str) -> str:
    parts = heading.split('{')

    title = parts[0].strip()

    # Take the second part and strip out spaces, '#', and '}'
    anchor_id = parts[1].strip(' #}') if len(parts) > 1 else None
    return (title, anchor_id)

def make_url(doc_id: str, anchor: str | None) -> str:
    without_docs = doc_id.removeprefix('docs/')
    without_md = without_docs.removesuffix('.md')
    remove_suffix = without_md.removesuffix('index')
    

    return 'https://fastapi.triangolo.com/' + remove_suffix + (anchor if anchor else '')

def chunks(file_dir):
    chunk_texts = []
    file_chunks = chunk_file(file_dir)
    for i, chunk in enumerate(file_chunks):
        metadata = chunk.metadata
        pairs = [split_anchor(chunk.metadata[k]) for k in ('h1', 'h2', 'h3') if k in chunk.metadata]
        heading_path = [text for text, _ in pairs]
        anchors = [a for _, a in pairs if a]
        anchor = anchors[-1] if anchors else None
        original_text = chunk.page_content
        doc_id = str(file_dir.relative_to('data/raw'))
        title = heading_path[0] if len(heading_path) > 0 else str(file_dir).split('/')[-1]
        page_content = " > ".join(heading_path) + "\n\n" + original_text
        chunk_id = f"{doc_id}--{i:03d}"
        url = make_url(doc_id, anchor)
        doc = Document(
            page_content=page_content,
            metadata={
                "doc_id": doc_id,
                "title": title,
                "heading_path": " > ".join(heading_path),
                "anchor": anchor or "",
                "original_text": original_text,
                "chunk_id": chunk_id,
                "chunk_index": i,
                "url": url
            }
        )
        chunk_texts.append(doc)
    return chunk_texts

def load_and_chunk(data_dir: str = 'data/raw') -> list[Document]:
    return [d for f in find_files(data_dir) for d in chunks(f)]
