# CLAUDE.md: Requirements Generator Project Guidelines

## Build & Run Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run streamlit_app.py

# Debug mode
streamlit run --debug streamlit_app.py

# Deploy to Streamlit Cloud
# The app is currently deployed at: https://reqgen.streamlit.app/
```

## Code Style Guidelines
- **Imports**: Group imports by standard lib, third-party, then local modules with blank lines between groups
- **Formatting**: PEP 8 conventions with 4-space indentation
- **Types**: Use type hints where possible, especially for function parameters and return values
- **Naming**: 
  - snake_case for variables, functions, and methods
  - CamelCase for classes
  - ALL_CAPS for constants
- **Error Handling**: Use try/except blocks with specific exception types and informative error messages
- **Documentation**: Docstrings for functions and classes, following Google docstring format
- **Streamlit State**: Manage state using st.session_state, initialize early with default values
- **Chains**: Follow LangChain conventions for creating and connecting chains