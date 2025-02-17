# Requirements Generator

A Streamlit-based application that leverages AI to generate, validate, and refine software requirements specifications. The system uses your preferred LLM to create comprehensive requirements documents with continuous validation and improvement capabilities.

The application can be found here: https://reqgen.streamlit.app/

## Features

- **Intelligent Requirements Generation**: Automatically generates structured requirements across multiple categories:
  - Functional Requirements
  - Non-Functional Requirements
  - Technical Requirements
  - User Interface Requirements
  - Security Requirements

- **Real-time Validation**: Analyses requirements for:
  - Ambiguities
  - Inconsistencies
  - Missing Elements
  - Potential Conflicts

- **Quality Metrics**: Tracks and displays:
  - Overall Quality Score
  - Issue Counts by Category
  - Progress Over Iterations

- **Monitoring**: Portkey to monitor LLM calls:
  - Track user requests for diagnostics
  - Track costs

- **Interactive Refinement**: Allows iterative improvement of requirements based on validation feedback

- **Version History**: Maintains a complete history of requirement iterations and changes

- **Export Functionality**: Download complete requirements documents in Markdown format

## Prerequisites

- Python 3.7+
- Streamlit
- LangChain
- Portkey AI

## Environment Variables

The following environment variables need to be set:

```
PORTKEY_API_KEY=your_portkey_api_key
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd requirements-generator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables in `.streamlit/secrets.toml`:
```toml
PORTKEY_API_KEY = "your_portkey_api_key"
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run streamlit_app.py
```

2. Access the application in your web browser (typically at `http://localhost:8501`)

3. Enter your project details:
   - Provide a project description
   - Add any additional context or constraints
   - Click "Generate Initial" to create the first set of requirements

4. Review and refine:
   - Examine the generated requirements
   - Review validation feedback
   - Click "Refine" to improve the requirements based on feedback
   - Repeat until satisfied with the quality score

5. Export your requirements:
   - Use the "Download Requirements Document" button to save your work

## Project Structure

```
requirements-generator/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── secrets.toml      # Configuration secrets
└── README.md             # This file
```

## Dependencies

- `streamlit`: Web application framework
- `langchain-openai`: OpenAI integration for LangChain
- `portkey-ai`: API gateway and management
- `uuid`: Unique identifier generation
- `json`: JSON parsing and handling

## Configuration

The application uses the following default settings:

- Temperature: 0.7 (for AI generation)
- Portkey configuration: 'pc-google-5fbd96'
- Session-based tracking for consistent results

## Quality Scoring

Quality scores are calculated based on the following deductions:
- Ambiguity: -5 points
- Inconsistency: -8 points
- Missing Element: -10 points
- Conflict: -12 points

Starting score is 100, and the final score is capped between 0 and 100.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Specify your license here]

## Support

None. You're on your own.