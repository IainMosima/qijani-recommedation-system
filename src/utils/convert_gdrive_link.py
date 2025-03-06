import re

def convert_gdrive_link(url: str) -> str:
    """
    Converts a public Google Drive share link to a direct download link.
    
    Example:
    Input: "https://drive.google.com/file/d/1ntkwGeHmmkyQXtEuP3yIC_zfeDbyVvNt/view?usp=sharing"
    Output: "https://drive.google.com/uc?export=download&id=1ntkwGeHmmkyQXtEuP3yIC_zfeDbyVvNt"
    
    If the URL is not recognized, returns the original URL.
    """
    # Try to extract the file ID from the standard share URL
    match = re.search(r'/d/([^/]+)', url)
    if match:
        file_id = match.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    
    # Fallback: check for an id query parameter
    match = re.search(r'id=([^&]+)', url)
    if match:
        file_id = match.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    
    return url