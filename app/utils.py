import re

def enhanced_text_cleaning(text: str) -> str:
    """Cleans text by removing special characters, extra whitespace, and leftover JS code.
       Also handles common HTML entities and potential multi-line noise."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\([^)]*\)\.push\({}\);', '', text) # Remove JS artifacts
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'<.*?>', '', text) # Remove HTML tags
    text = re.sub(r'&[a-zA-Z0-9#]+;', '', text) # Remove HTML entities like &nbsp;
    text = re.sub(r'[\r\n]+', ' ', text) # Replace multiple newlines with single space
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', '', text) # Keep only safe characters
    text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
    return text