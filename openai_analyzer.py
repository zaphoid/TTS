import re
import json
import openai
import time
from typing import List, Tuple

class OpenAIEnhancedAnalyzer:
    """Text analyzer that uses OpenAI's API to accurately classify text segments"""
    
    def __init__(self, api_key=None):
        """Initialize with the same API key used for TTS"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key
        
    def classify_paragraph(self, paragraph: str) -> str:
        """Use OpenAI to classify the paragraph type based on content"""
        if not paragraph.strip():
            return "narrator"
            
        try:
            # Skip API call for obvious cases to save tokens
            if self._is_clearly_code(paragraph):
                return "computer"
                
            # Create a prompt for the OpenAI API to classify the text
            prompt = [
                {"role": "system", "content": "You are a text classifier that categorizes paragraphs into three types: 'narrator', 'dialog', or 'computer'. 'narrator' is for narrative text, descriptions, and the protagonist's thoughts. 'dialog' is for conversations, character speech, and messages between characters. 'computer' is for code snippets, technical descriptions, terminal outputs, file contents, and UI elements."},
                {"role": "user", "content": f"Classify the following paragraph as exactly one of 'narrator', 'dialog', or 'computer' without explanation. Just return the single word classification.\n\nParagraph: {paragraph}"}
            ]
            
            # Make API call to OpenAI
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",  # Use a simpler/cheaper model for classification
                messages=prompt,
                max_tokens=10,  # We only need one word
                temperature=0.1  # Low temperature for consistent results
            )
            
            # Get the classification result
            result = response.choices[0].message.content.strip().lower()
            
            # Handle any unexpected responses
            if result not in ["narrator", "dialog", "computer"]:
                # If unclear, make a reasonable guess based on text properties
                if '"' in paragraph and paragraph.count('"') >= 2:
                    return "dialog"
                elif any(code_marker in paragraph for code_marker in ["```", "def ", "class ", "import ", "function"]):
                    return "computer"
                else:
                    return "narrator"
                    
            return result
            
        except Exception as e:
            print(f"Error in OpenAI classification: {str(e)}")
            # Fallback to simple pattern matching if API fails
            return self._fallback_classify(paragraph)
    
    def _is_clearly_code(self, paragraph: str) -> bool:
        """Quick check for obvious code/technical content to save API calls"""
        code_patterns = [
            r"```\w*\n",  # Code block markers with language
            r"import\s+[a-zA-Z_]+",  # Python imports
            r"def\s+[a-zA-Z_]+\(",  # Function definitions
            r"class\s+[a-zA-Z_]+",  # Class definitions
            r"function\s+[a-zA-Z_]+\(",  # JavaScript functions
            r"<[a-zA-Z]+>.*</[a-zA-Z]+>",  # HTML tags
            r"\$\(.*\)",  # jQuery selectors
            r"@\w+\s*\(",  # Python decorators
            r"#include\s+<.*>",  # C/C++ includes
            r"package\s+[a-zA-Z_]+;",  # Java package declarations
            r"from\s+[a-zA-Z_\.]+\s+import",  # Python from imports
            r"for\s*\(\s*\w+\s*=\s*\d+;",  # C-style for loops
            r"const\s+[a-zA-Z_]+\s*=",  # JavaScript const
            r"let\s+[a-zA-Z_]+\s*=",  # JavaScript let
            r"var\s+[a-zA-Z_]+\s*=",  # JavaScript var
            r"public\s+(static\s+)?(void|int|String)",  # Java method declarations
        ]
        
        for pattern in code_patterns:
            if re.search(pattern, paragraph):
                return True
                
        # Check for structured data patterns
        if (paragraph.strip().startswith("{") and paragraph.strip().endswith("}")) or \
           (paragraph.strip().startswith("[") and paragraph.strip().endswith("]")):
            return True
            
        return False
        
    def _fallback_classify(self, paragraph: str) -> str:
        """Simple pattern-based classification as fallback"""
        # Check for code/computer content
        if (paragraph.strip().startswith('```') or 
            paragraph.strip().startswith('>') or 
            paragraph.strip().startswith('File:') or
            re.search(r'Filename:|Log:|Author:|Last Modified', paragraph)):
            return "computer"
            
        # Check for dialogue
        dialogue_matches = re.findall(r'"[^"]*"', paragraph)
        if dialogue_matches and len(''.join(dialogue_matches)) > len(paragraph) * 0.3:
            return "dialog"
            
        # Default to narrator
        return "narrator"
    
    def analyze_text(self, full_text: str) -> List[Tuple[str, str]]:
        """Split text into segments and assign types using OpenAI API"""
        segments = []
        
        # Split by paragraphs (respecting blank lines)
        paragraphs = re.split(r'\n\s*\n', full_text)
        
        # Batch process to minimize API calls
        # Process every 5th paragraph with OpenAI and use pattern matching for others
        paragraph_types = []
        
        print("Beginning text analysis with OpenAI...")
        for i, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                paragraph_types.append(None)
                continue
                
            # API call with rate limiting
            try:
                # Use API for longer paragraphs or every 5th paragraph
                if i % 5 == 0 or len(paragraph) > 500:
                    paragraph_type = self.classify_paragraph(paragraph)
                    print(f"Paragraph {i}: Classified as '{paragraph_type}' (via API)")
                else:
                    paragraph_type = self._fallback_classify(paragraph)
                    print(f"Paragraph {i}: Classified as '{paragraph_type}' (via pattern)")
                
                paragraph_types.append(paragraph_type)
                
                # Add some delay to avoid rate limiting
                if i % 5 == 0:
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"Error processing paragraph {i}: {str(e)}")
                paragraph_types.append("narrator")  # Default to narrator on error
        
        # Create segments with classified types
        for i, paragraph in enumerate(paragraphs):
            if not paragraph.strip() or paragraph_types[i] is None:
                continue
                
            paragraph_type = paragraph_types[i]
            
            # For longer paragraphs, check if we need to further split
            if len(paragraph) > 1000:
                # Split by sentences for long paragraphs
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) > 1000:
                        if current_chunk:
                            segments.append((paragraph_type, current_chunk.strip()))
                            current_chunk = sentence
                        else:
                            segments.append((paragraph_type, sentence.strip()))
                    else:
                        current_chunk += " " + sentence if current_chunk else sentence
                
                if current_chunk:
                    segments.append((paragraph_type, current_chunk.strip()))
            else:
                segments.append((paragraph_type, paragraph.strip()))
        
        return segments
        
    def batch_analyze(self, full_text: str, batch_size: int = 10) -> List[Tuple[str, str]]:
        """Analyze text in batches for more efficient API usage"""
        segments = []
        
        # Split by paragraphs (respecting blank lines)
        paragraphs = re.split(r'\n\s*\n', full_text)
        valid_paragraphs = [p for p in paragraphs if p.strip()]
        
        # Prepare batches
        batches = [valid_paragraphs[i:i+batch_size] for i in range(0, len(valid_paragraphs), batch_size)]
        
        for batch_idx, batch in enumerate(batches):
            print(f"Processing batch {batch_idx+1}/{len(batches)}...")
            
            # Create a combined prompt for multiple paragraphs
            prompt = [
                {"role": "system", "content": "You are a text classifier that categorizes paragraphs into three types: 'narrator', 'dialog', or 'computer'. For each paragraph provided, classify it as one of these categories. 'narrator' is for narrative text, descriptions, and the protagonist's thoughts. 'dialog' is for conversations, character speech, and messages between characters. 'computer' is for code snippets, technical descriptions, terminal outputs, file contents, and UI elements."},
                {"role": "user", "content": "Classify each of the following paragraphs as either 'narrator', 'dialog', or 'computer'. Return your answer as a JSON array with ONE classification per paragraph, like this: [\"narrator\", \"dialog\", \"computer\", ...]. No explanations, just the array of classifications.\n\n" + "\n\nPARAGRAPH:\n".join([f"{i+1}. {p}" for i, p in enumerate(batch)])}
            ]
            
            try:
                # Make API call to OpenAI
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=prompt,
                    max_tokens=100,
                    temperature=0.1
                )
                
                # Get the classification results
                result = response.choices[0].message.content.strip()
                
                # Try to parse as JSON
                try:
                    classifications = json.loads(result)
                    
                    # Validate classifications
                    if isinstance(classifications, list) and len(classifications) == len(batch):
                        # Add each paragraph with its classification
                        for i, paragraph in enumerate(batch):
                            classification = classifications[i].lower() if classifications[i].lower() in ["narrator", "dialog", "computer"] else "narrator"
                            
                            # For longer paragraphs, split if needed
                            if len(paragraph) > 1000:
                                # Split by sentences
                                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                                current_chunk = ""
                                
                                for sentence in sentences:
                                    if len(current_chunk) + len(sentence) > 1000:
                                        if current_chunk:
                                            segments.append((classification, current_chunk.strip()))
                                            current_chunk = sentence
                                        else:
                                            segments.append((classification, sentence.strip()))
                                    else:
                                        current_chunk += " " + sentence if current_chunk else sentence
                                
                                if current_chunk:
                                    segments.append((classification, current_chunk.strip()))
                            else:
                                segments.append((classification, paragraph.strip()))
                    else:
                        # Fallback to individual classification
                        for paragraph in batch:
                            classification = self._fallback_classify(paragraph)
                            segments.append((classification, paragraph.strip()))
                            
                except json.JSONDecodeError:
                    # If JSON parsing fails, use fallback classification
                    print("Failed to parse API response as JSON, using fallback classification")
                    for paragraph in batch:
                        classification = self._fallback_classify(paragraph)
                        segments.append((classification, paragraph.strip()))
                
                # Add delay between batches
                if batch_idx < len(batches) - 1:
                    time.sleep(1)
                    
            except Exception as e:
                print(f"Error processing batch {batch_idx+1}: {str(e)}")
                # Fallback to simple classification
                for paragraph in batch:
                    classification = self._fallback_classify(paragraph)
                    segments.append((classification, paragraph.strip()))
        
        return segments

# Integration with the EnhancedTTSConverter class
# This would replace the existing TextAnalyzer in your code

def get_openai_enhanced_converter(api_key=None):
    """Creates an enhanced converter that uses OpenAI for text classification"""
    converter = EnhancedTTSConverter()
    converter.analyzer = OpenAIEnhancedAnalyzer(api_key)
    return converter
