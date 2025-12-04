"""Related work profiler for extracting dataset context from research papers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

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
                - summary: Extracted summary about the dataset
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
    def analyze_paper(
        self,
        pdf_path: str,
        dataset_name: str,
        extraction_prompt: Optional[str] = None,
        max_pages: Optional[int] = None,
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
        
        # Step 2: Extract profile using LLM
        profile = self.extract_related_profile(
            paper_text=paper_text,
            dataset_name=dataset_name,
            extraction_prompt=extraction_prompt
        )
        
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