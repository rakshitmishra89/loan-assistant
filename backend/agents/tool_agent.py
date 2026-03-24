from backend.adapters import tools_adapter

def process(extracted_data: dict) -> dict:
    """Passes the cleaned numbers to the math tools."""
    
    # Run the tools adapter we updated earlier
    results = tools_adapter.run_all(extracted_data)
    
    return results