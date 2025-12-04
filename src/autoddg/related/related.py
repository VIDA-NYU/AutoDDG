"""Related work profiler for extracting dataset context from research papers."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Dict, List, Any

import yaml
from beartype import beartype
from openai import OpenAI
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


@beartype
class RelatedWorkProfiler:
    """
    Extracts dataset context and information from related research papers.
    
    This profiler analyzes research papers (PDFs) to extract contextual information
    about datasets, including their characteristics, usage, and provenance.
    """
    
    def __init__(
        self,
        client: OpenAI,
        model_name: str = "gpt-4o-mini",
        prompts_config: Optional[dict] = None,
    ) -> None:
        """
        Initialize the RelatedWorkProfiler.
        
        Args:
            client: OpenAI client instance for LLM calls
            model_name: Name of the model to use for extraction
            prompts_config: Dictionary containing prompts configuration. If None, loads from prompts.yaml
        """
        self.client = client
        self.model_name = model_name
        
        # Load prompts from config
        if prompts_config is None:
            prompts_config = self._load_prompts_config()
        
        self.prompts = prompts_config.get("related_work_extraction", {})
        self.default_extraction_prompt = self.prompts.get("default_prompt", "")
        self.system_message = self.prompts.get("system_message", "You are an expert academic research assistant.")
    
    def _load_prompts_config(self) -> dict:
        """
        Load prompts configuration from prompts.yaml file.
        
        Returns:
            Dictionary containing prompts configuration
        """
        # Try to find prompts.yaml in the autoddg package
        try:
            from importlib.resources import files
            prompts_path = files("autoddg.configurations").joinpath("prompts.yaml")
            with prompts_path.open("r") as f:
                return yaml.safe_load(f)
        except (ImportError, FileNotFoundError):
            # Fallback: try relative path
            # We're in autoddg/related/related.py, need to go to autoddg/configurations/
            current_dir = Path(__file__).parent  # autoddg/related/
            autoddg_dir = current_dir.parent      # autoddg/
            prompts_path = autoddg_dir / "configurations" / "prompts.yaml"
            
            if prompts_path.exists():
                with open(prompts_path, "r") as f:
                    return yaml.safe_load(f)
            else:
                # Return empty dict if no config found
                print(f"Warning: prompts.yaml not found at {prompts_path}, using empty config")
                return {}
    
    @beartype
    def chunk_text(
        self,
        paper_text: str,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
    ) -> list[str]:
        """
        Splits the full paper text into context-preserving chunks.
        
        Args:
            paper_text: The full text content of the research paper.
            chunk_size: The desired maximum size of each chunk (in characters).
            chunk_overlap: The number of characters to overlap between adjacent chunks.
            
        Returns:
            A list of text strings (chunks).
        """
        # Use standard academic separators to preserve paragraphs and sentences
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",  # Try to split by paragraph first
                "\n",    # Then by newline
                " ",     # Then by space
                "",      # Fallback by character
            ],
            length_function=len,
            is_separator_regex=False,
        )
        
        chunks = splitter.split_text(paper_text)
        print(f"Original text split into {len(chunks)} chunks.")
        return chunks
    
    def find_anchor_chunks(
        self,
        chunks: List[str],
        dataset_name: str,
        min_tokens_to_match: int = 2
    ) -> List[int]:
        """
        Searches a list of text chunks for references to a given dataset name.

        This function implements a semi-broad search strategy:
        1. Splits the dataset name into key tokens (excluding common words).
        2. Requires a minimum number of these key tokens to be present in a chunk.
        3. The search is case-insensitive.

        Returns:
            A list of IDs or indices of the chunks that contain enough matching tokens.
        """
        # 1. Pre-process the dataset name to get key search terms
        # Define common stop words to ignore (can be expanded)
        stop_words = {'the', 'a', 'an', 'database', 'data', 'of', 'and', 'for', 'in', 'to', 'with'}
        
        # Split the name into tokens, filter out stop words, and convert to lowercase
        key_tokens = set(
            re.findall(r'\b\w+\b', dataset_name.lower())
        ) - stop_words

        if not key_tokens:
            print("Warning: Dataset name contains only stop words after filtering. Cannot perform robust search.")
            # Fallback to searching the full, non-processed name
            key_tokens = {dataset_name.lower()}
            # If we use the full name, we must match at least 1 token
            min_tokens_to_match = 1
        
        # Ensure min_tokens_to_match is not more than the number of key tokens
        min_tokens_to_match = min(min_tokens_to_match, len(key_tokens))
        
        # If a short name like 'FluPRINT' is used, require matching all tokens
        if len(key_tokens) < min_tokens_to_match:
            min_tokens_to_match = len(key_tokens)
        
        # 2. Search each chunk
        anchor_chunk_ids = []
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.lower()
            
            # Identify which key tokens are present in the current chunk
            matched_tokens = 0
            
            for token in key_tokens:
                # Check for the token as a whole word boundary match
                if re.search(r'\b' + re.escape(token) + r'\b', chunk_text):
                    matched_tokens += 1
                # print(f"looking for {token} in {chunk_text} \n")
                    
            # 3. Apply the matching threshold
            if matched_tokens >= min_tokens_to_match:
                # We use 'id' if available, otherwise the index 'i'
                chunk_identifier = i
                anchor_chunk_ids.append(chunk_identifier)

        return anchor_chunk_ids
    
    def get_logical_context_blocks(
        self,
        all_chunks: List[str],
        anchor_chunk_ids: List[int],
        context_window_size: int = 2
    ) -> List[str]:
        """
        Creates coherent, logical context blocks by merging adjacent anchor chunks 
        and expanding the context window around non-adjacent ones.
        """
        
        # 1. Sort and ensure uniqueness
        sorted_anchor_ids = sorted(list(set(anchor_chunk_ids)))
        
        # 2. Identify all indices to include in the final context
        context_indices_to_include = set()
        num_chunks = len(all_chunks)
        
        # Iterate through anchor chunks to apply merging/expansion
        for anchor_id in sorted_anchor_ids:
            
            # Skip if this chunk is already part of a previous block's context
            if anchor_id in context_indices_to_include:
                continue
                
            # Determine the boundaries for this logical block (expansion)
            # Start by expanding the context around the anchor
            start_index = max(0, anchor_id - context_window_size)
            end_index = min(num_chunks - 1, anchor_id + context_window_size)

            # Extend the end_index if the next chunks are also anchors (merging)
            current_id = anchor_id + 1
            while current_id < num_chunks and current_id in sorted_anchor_ids:
                # Anchor chunk is adjacent, so include it and expand the end index
                end_index = min(num_chunks - 1, current_id + context_window_size)
                current_id += 1
                
            # Add all unique indices in this block's range to the set
            for i in range(start_index, end_index + 1):
                context_indices_to_include.add(i)

        # 3. Create the final logical blocks
        # We must merge all the contiguous index ranges into full text blocks
        final_logical_blocks = []
        
        # Convert set to sorted list for easy iteration
        sorted_indices = sorted(list(context_indices_to_include))
        
        if not sorted_indices:
            return []
            
        current_block_chunks = []
        for i, idx in enumerate(sorted_indices):
            is_contiguous = (i == 0) or (idx == sorted_indices[i-1] + 1)
            
            if is_contiguous:
                # Continue the current block
                current_block_chunks.append(all_chunks[idx])
            else:
                # Non-contiguous, finalize the previous block and start a new one
                final_logical_blocks.append("\n\n".join(current_block_chunks))
                current_block_chunks = [all_chunks[idx]]

        # Add the last block
        if current_block_chunks:
            final_logical_blocks.append("\n\n".join(current_block_chunks))
            
        print(f"Reduced to {len(final_logical_blocks)} logical context blocks.")
        return final_logical_blocks

    @beartype
    def extract_text_from_pdf(
        self,
        pdf_path: str,
        max_pages: Optional[int] = None
    ) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            max_pages: Optional limit on number of pages to extract
            
        Returns:
            Extracted text content from the PDF
            
        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            Exception: If there's an error reading the PDF
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")
        
        print(f"Reading PDF from: {pdf_path}")
        
        try:
            with open(pdf_path, "rb") as pdf_file:
                reader = PdfReader(pdf_file)
                total_pages = len(reader.pages)
                pages_to_read = min(max_pages, total_pages) if max_pages else total_pages
                
                paper_text = ""
                for i in range(pages_to_read):
                    page = reader.pages[i]
                    paper_text += page.extract_text(extraction_mode="plain") + "\n\n"
                
                print(f"Successfully extracted text from {pages_to_read} pages (total: {total_pages} pages)")
                print(f"Total characters extracted: {len(paper_text)}")
                
                return paper_text
                
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {e}")
    
    @beartype
    def extract_related_profile(
        self,
        paper_text: str,
        dataset_name: str,
        extraction_prompt: Optional[str] = None,
    ) -> dict:
        """
        Extract related work profile from paper text using LLM.
        
        Args:
            paper_text: Full text content of the research paper
            dataset_name: Name of the dataset to focus extraction on
            extraction_prompt: Custom extraction prompt. If None, uses default.
                              Use {paper_text} and {dataset_name} as placeholders.
            
        Returns:
            Dictionary containing the related work profile with keys:
                - summary: Extracted summary about the datraset
                - dataset_name: Name of the dataset
                - source_length: Character count of source paper
        """
        # Use custom prompt if provided, otherwise use default
        prompt_template = extraction_prompt if extraction_prompt else self.default_extraction_prompt
        
        # Format the prompt with paper text and dataset name
        formatted_prompt = prompt_template.format(
            paper_text=paper_text,
            dataset_name=dataset_name
        )
        
        print(f"Extracting related work profile for dataset: {dataset_name}")
        print(f"Sending {len(formatted_prompt)} characters to LLM...")
        
        # Call the LLM
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_message
                    },
                    {
                        "role": "user",
                        "content": formatted_prompt
                    }
                ],
                temperature=0.1,
                # response_format={"type": "json_object"}
            )
            
            summary = response.choices[0].message.content.strip()
            
            print(f"Successfully extracted profile ({len(summary)} characters)")
            
            return {
                "summary": summary,
                "dataset_name": dataset_name,
                "source_length": len(paper_text)
            }
            
        except Exception as e:
            raise Exception(f"Error calling LLM for extraction: {e}")
        
    @beartype
    def _extract_profile_from_context(
        self,
        context_blocks: List[str],  # New parameter: list of relevant context blocks
        dataset_name: str,
        extraction_prompt: Optional[str] = None,
    ) -> dict:
        """
        Extract related work profile from selected context blocks using LLM.
        """
        
        # 1. Combine all logical context blocks into a single string
        # Use a clear separator so the LLM knows where one block ends and the next begins
        combined_context = "\n\n--- LOGICAL BLOCK SEPARATOR ---\n\n".join(context_blocks)
        
        # Use custom prompt if provided, otherwise use default
        prompt_template = extraction_prompt if extraction_prompt else self.default_extraction_prompt
        
        # Format the prompt with the combined context and dataset name
        formatted_prompt = prompt_template.format(
            paper_text=combined_context,  # paper_text now refers to the combined, relevant context
            dataset_name=dataset_name
        )
        
        print(f"Extracting profile for dataset: {dataset_name}")
        print(f"Sending {len(formatted_prompt)} characters of CONTEXT to LLM...")
        
        # Call the LLM (rest of the code remains the same)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_message
                    },
                    {
                        "role": "user",
                        "content": formatted_prompt
                    }
                ],
                temperature=0.1,
                # response_format={"type": "json_object"}
            )
            
            summary = response.choices[0].message.content.strip()
            
            print(f"Successfully extracted profile ({len(summary)} characters)")
            
            return {
                "summary": summary,
                "dataset_name": dataset_name,
                "source_length": len(combined_context) # Source length is now the context size
            }
            
        except Exception as e:
            raise Exception(f"Error calling LLM for extraction: {e}")
    
    # Insert this method into your RelatedWorkProfiler class

    def score_chunk_relevance(self, chunk_text: str, dataset_name: str) -> bool:
        """Uses LLM to determine if a chunk is semantically relevant to the dataset."""
        
        # Use a lightweight model for this scoring task
        # IMPORTANT: Make sure this model is suitable for JSON or boolean output
        scoring_model = self.model_name # You can choose a faster/cheaper model here if desired

        prompt = f"""
        Analyze the following text chunk from a research paper. The focus is on the dataset named '{dataset_name}'.
        
        Chunk:
        ---
        {chunk_text}
        ---
        
        Is this chunk semantically relevant to the dataset? Relevance means it discusses the dataset's use, characteristics, results, or limitations. It is NOT relevant if it is a list of references, acknowledgments, or a general background statement.
        
        Respond with only a single word: YES or NO.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=scoring_model,
                messages=[
                    {"role": "system", "content": "You are a text relevance classifier. Respond only with 'YES' or 'NO'."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
            )
            
            response_content = response.choices[0].message.content.strip().upper()
            
            # Check for explicit YES response
            return response_content == "YES"
            
        except Exception as e:
            # Fallback in case of API error
            print(f"Error scoring chunk: {e}. Defaulting to non-relevant.")
            return False
        
    @beartype
    def analyze_paper(
        self,
        pdf_path: str,
        dataset_name: str,
        extraction_prompt: Optional[str] = None,
        max_pages: Optional[int] = None,
        chunk_size: int = 2000, 
        chunk_overlap: int = 200,
        context_window_size: int = 3
    ) -> dict:
        """
        Complete pipeline: Extract text from PDF and generate related work profile.
        
        Args:
            pdf_path: Path to the PDF file
            dataset_name: Name of the dataset to focus extraction on
            extraction_prompt: Custom extraction prompt. If None, uses default.
            max_pages: Optional limit on number of pages to extract
            
        Returns:
            Dictionary containing the related work profile
        """
        # Step 1: Extract text from PDF
        paper_text = self.extract_text_from_pdf(pdf_path, max_pages=max_pages)

        original_chunks = self.chunk_text(
            paper_text=paper_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        relevant_chunks: List[str] = []
        
        for i, chunk in enumerate(original_chunks):
            # Call the new LLM rater method
            if self.score_chunk_relevance(chunk_text=chunk, dataset_name=dataset_name):
                relevant_chunks.append(chunk)
                
        # Use the filtered chunks as the context blocks
        logical_context_blocks = relevant_chunks
        
        # anchor_ids = self.find_anchor_chunks(
        #     chunks=original_chunks,
        #     dataset_name=dataset_name,
        #     min_tokens_to_match=2 # Use your existing robust logic
        # )

        # logical_context_blocks = self.get_logical_context_blocks(
        #     all_chunks=original_chunks,
        #     anchor_chunk_ids=anchor_ids,
        #     context_window_size=context_window_size
        # )

        if not logical_context_blocks:
            print("Warning: No relevant chunks found. Falling back to using the full text.")
            logical_context_blocks = [paper_text]
        
        # Step 2: Extract profile using LLM
        profile = self._extract_profile_from_context( # Call the new helper method
                    context_blocks=logical_context_blocks,
                    dataset_name=dataset_name,
                    extraction_prompt=extraction_prompt
                )        
        profile["full_source_length"] = len(paper_text)
        return profile


# Example usage for testing in notebook
if __name__ == "__main__":
    # This block can be used for testing
    print("RelatedWorkProfiler class loaded successfully!")
    print("\nExample usage:")
    print("""
from openai import OpenAI
from related_work import RelatedWorkProfiler

# Initialize client
client = OpenAI(api_key="your-api-key")

# Create profiler
profiler = RelatedWorkProfiler(client=client, model_name="gpt-4o-mini")

# Analyze a paper
profile = profiler.analyze_paper(
    pdf_path="path/to/paper.pdf",
    dataset_name="Your Dataset Name"
)

print(profile["summary"])
""")