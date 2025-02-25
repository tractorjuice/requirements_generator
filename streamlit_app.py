"""
Requirements Generator - A Streamlit application for generating and refining software requirements.

This application leverages LLMs to create, validate, and improve software requirements
specifications based on project descriptions and context.
"""

import streamlit as st
import uuid, json
from langchain_openai import ChatOpenAI
from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Page configuration
st.set_page_config(page_title="Requirements Generator", layout="wide")

# Initialize session state variables
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'history' not in st.session_state:
    st.session_state.history = []
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'current_requirements' not in st.session_state:
    st.session_state.current_requirements = None
if 'current_validation' not in st.session_state:
    st.session_state.current_validation = None
if 'iteration_count' not in st.session_state:
    st.session_state.iteration_count = 0
if 'quality_score' not in st.session_state:
    st.session_state.quality_score = 0
if 'validation_metrics' not in st.session_state:
    st.session_state.validation_metrics = {
        'ambiguities': 0,
        'inconsistencies': 0,
        'missing_elements': 0,
        'conflicts': 0
    }

# Application configuration parameters
portkey_api_key = st.secrets["PORTKEY_API_KEY"]
openai_api_key = st.secrets["OPENAI_API_KEY"]
google_portkey_config = st.secrets["google_portkey_config"]
tags = ["Requirements Generator"]
temperature = 0.7

@st.cache_resource
def get_portkey_headers():
    """
    Create and cache Portkey headers for API requests.
    
    Returns:
        dict: Headers for Portkey API requests with tracking info
    """
    return createHeaders(
        api_key=portkey_api_key,
        config=google_portkey_config,
        trace_id=st.session_state.session_id,
        metadata={"_user": st.session_state.session_id}
    )

def init_llm():
    """
    Initialize the LLM with Portkey configuration.
    
    Returns:
        ChatOpenAI: Configured LangChain ChatOpenAI instance
    """
    portkey_headers = get_portkey_headers()
    return ChatOpenAI(
        base_url=PORTKEY_GATEWAY_URL,
        default_headers=portkey_headers,
        api_key="PORTKEY_API_KEY",
        temperature=temperature,
        tags=tags + [f"session_{st.session_state.session_id}"]
    )

def create_chains(llm):
    """
    Create LangChain processing chains for the requirements generation workflow.
    
    Args:
        llm (ChatOpenAI): The language model instance
        
    Returns:
        tuple: Three chains for initial generation, validation, and refinement
    """
    # Initial requirements generation chain
    initial_req_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a requirements analysis expert. Generate detailed requirements based on the provided information.
        Break down the requirements into:
        1. Functional Requirements
        2. Non-Functional Requirements
        3. Technical Requirements
        4. User Interface Requirements
        5. Security Requirements

        Format each requirement clearly and number them within their categories."""),
        ("human", "Project Information: {project_info}\n\nAdditional Context: {context}\n\nGenerate comprehensive requirements:")
    ])
    initial_req_chain = initial_req_prompt | llm | StrOutputParser()

    # Validation chain
    validation_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a requirements validation expert. Review the generated requirements and identify:
        - Ambiguities
        - Inconsistencies
        - Missing critical elements
        - Potential conflicts

        For each issue identified, provide specific recommendations for improvement.
        Number each issue and recommendation clearly."""),
        ("human", "Requirements to validate: {requirements}\n\nProvide validation feedback:")
    ])
    validation_chain = validation_prompt | llm | StrOutputParser()

    # Refinement chain
    refinement_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a requirements refinement expert. Update the existing requirements based on the validation feedback.
        Maintain the same structure but incorporate the suggested improvements.
        Ensure all requirements remain clear, specific, and testable.
        Mark new or significantly modified requirements with [Updated] prefix."""),
        ("human", """Original Requirements: {requirements}

        Validation Feedback: {validation}

        Previous Context: {context}

        Generate refined requirements:""")
    ])
    refinement_chain = refinement_prompt | llm | StrOutputParser()

    return initial_req_chain, validation_chain, refinement_chain

def analyze_validation_feedback(validation_text):
    """
    Analyze validation feedback to extract metrics and calculate quality score.
    
    Args:
        validation_text (str): The validation feedback text to analyze
        
    Returns:
        dict: Quality score and metrics including counts of issues by type
    """
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a JSON-only metrics analyzer. Output MUST be a single valid JSON object with ABSOLUTELY NO OTHER TEXT.
DO NOT include markdown formatting, code blocks, or any other decorations.
ONLY return the raw JSON object.

Example of exact format you must follow:
{{
    "quality_score": 85,
    "metrics": {{
        "ambiguities": 2,
        "inconsistencies": 1,
        "missing_elements": 3,
        "conflicts": 0
    }}
}}

Rules for analysis:
- Count each instance of issues mentioned in the validation feedback
- Calculate quality score starting at 100 and subtracting:
  * -5 for each ambiguity
  * -8 for each inconsistency
  * -10 for each missing element
  * -12 for each conflict
- Ensure score stays between 0 and 100
- Return ONLY the raw JSON object, no markdown, no code blocks"""),
        ("human", "Count the issues in this validation feedback and generate metrics ONLY as a JSON object: {validation}")
    ])

    analysis_chain = analysis_prompt | st.session_state.llm | StrOutputParser()

    try:
        # Get the analysis result
        analysis_result = analysis_chain.invoke({"validation": validation_text})

        # Clean the response to ensure proper JSON format
        # Remove markdown code block syntax if present
        analysis_result = analysis_result.replace("```json", "").replace("```", "")

        # Remove any leading/trailing whitespace
        analysis_result = analysis_result.strip()

        # Basic validation before parsing
        if not (analysis_result.startswith('{') and analysis_result.endswith('}')):
            st.write("Response does not start and end with curly braces")
            raise ValueError("Response is not in JSON format")

        # Parse the JSON response
        metrics = json.loads(analysis_result)

        # Validate and normalize the metrics data
        normalized_metrics = {
            "quality_score": max(0, min(100, int(metrics.get("quality_score", 0)))),
            "metrics": {
                "ambiguities": max(0, int(metrics.get("metrics", {}).get("ambiguities", 0))),
                "inconsistencies": max(0, int(metrics.get("metrics", {}).get("inconsistencies", 0))),
                "missing_elements": max(0, int(metrics.get("metrics", {}).get("missing_elements", 0))),
                "conflicts": max(0, int(metrics.get("metrics", {}).get("conflicts", 0)))
            }
        }

        return normalized_metrics

    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON format: {str(e)}")
        st.write("Failed to parse JSON:", analysis_result)
    except Exception as e:
        st.error(f"Error in analysis: {str(e)}")

    # Return default metrics if there's any error
    return {
        "quality_score": 0,
        "metrics": {
            "ambiguities": 0,
            "inconsistencies": 0,
            "missing_elements": 0,
            "conflicts": 0
        }
    }

# Initialize LLM if not already in session state
try:
    if st.session_state.llm is None:
        st.session_state.llm = init_llm()
except Exception as e:
    st.error(f"Error initializing components: {str(e)}")
    st.stop()

# Create LangChain processing chains
initial_req_chain, validation_chain, refinement_chain = create_chains(st.session_state.llm)

# Sidebar configuration for input and metrics
with st.sidebar:
    st.title("📋 Project Details")

    # Display progress metrics if available
    if st.session_state.quality_score > 0:
        st.subheader("Requirements Quality")
        st.progress(st.session_state.quality_score / 100, f"{st.session_state.quality_score}%")

        # Display metrics in a two-column layout
        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            st.metric("Ambiguities", st.session_state.validation_metrics['ambiguities'])
            st.metric("Inconsistencies", st.session_state.validation_metrics['inconsistencies'])
        with metrics_col2:
            st.metric("Missing Elements", st.session_state.validation_metrics['missing_elements'])
            st.metric("Conflicts", st.session_state.validation_metrics['conflicts'])

    st.divider()

    # Project description input
    project_info = st.text_area(
        "📝 Describe your project:",
        placeholder="e.g., A web-based inventory management system for a small retail business...",
        height=150
    )

    # Additional context input
    context = st.text_area(
        "🔍 Additional Context (optional):",
        placeholder="e.g., Target users, specific business rules, technical constraints...",
        height=100
    )

    # Action buttons in a two-column layout
    col1, col2 = st.columns(2)
    with col1:
        generate_button = st.button("Generate Initial", type="primary")
    with col2:
        refine_button = st.button("Refine", type="primary",
                                 disabled=not 'current_requirements' in st.session_state)

# Main content area
st.title("Requirements Generation System")

# Handle generate button click
if generate_button:
    if project_info:
        st.session_state.iteration_count = 0
        with st.spinner("Generating initial requirements..."):
            # Generate initial requirements
            initial_requirements = initial_req_chain.invoke({
                "project_info": project_info,
                "context": context or "No additional context provided."
            })
            st.session_state.current_requirements = initial_requirements

            # Generate validation feedback
            validation = validation_chain.invoke({
                "requirements": initial_requirements
            })
            st.session_state.current_validation = validation

            # Analyze validation feedback for metrics
            analysis = analyze_validation_feedback(validation)
            st.session_state.quality_score = analysis["quality_score"]
            st.session_state.validation_metrics = analysis["metrics"]

            # Store iteration in history
            st.session_state.history.append({
                "iteration": st.session_state.iteration_count,
                "project_info": project_info,
                "context": context,
                "requirements": initial_requirements,
                "validation": validation,
                "quality_score": st.session_state.quality_score,
                "metrics": st.session_state.validation_metrics
            })
    else:
        st.error("Please provide project information to generate requirements.")

# Handle refine button click
if refine_button:
    with st.spinner("Refining requirements..."):
        st.session_state.iteration_count += 1

        # Refine requirements based on validation feedback
        refined_requirements = refinement_chain.invoke({
            "requirements": st.session_state.current_requirements,
            "validation": st.session_state.current_validation,
            "context": context or "No additional context provided."
        })
        st.session_state.current_requirements = refined_requirements

        # Generate new validation feedback
        new_validation = validation_chain.invoke({
            "requirements": refined_requirements
        })
        st.session_state.current_validation = new_validation

        # Analyze updated validation feedback
        analysis = analyze_validation_feedback(new_validation)
        st.session_state.quality_score = analysis["quality_score"]
        st.session_state.validation_metrics = analysis["metrics"]

        # Store iteration in history
        st.session_state.history.append({
            "iteration": st.session_state.iteration_count,
            "project_info": project_info,
            "context": context,
            "requirements": refined_requirements,
            "validation": new_validation,
            "quality_score": st.session_state.quality_score,
            "metrics": st.session_state.validation_metrics
        })

# Display current requirements and validation feedback
if st.session_state.current_requirements:
    st.subheader(f"Current Requirements (Iteration {st.session_state.iteration_count})")

    # Display requirements and validation in two columns
    req_col, val_col = st.columns(2)

    with req_col:
        st.markdown("### 📋 Requirements")
        st.markdown(st.session_state.current_requirements)

    with val_col:
        st.markdown("### 🔍 Validation Feedback")
        st.markdown(st.session_state.current_validation)

# Display requirements history
if st.session_state.history:
    st.subheader("📚 Requirements History")
    for item in reversed(st.session_state.history):
        with st.expander(f"Iteration {item['iteration']}", expanded=False):
            st.markdown("**Requirements:**")
            st.write(item['requirements'])
            st.markdown("**Validation Feedback:**")
            st.write(item['validation'])
            if item['context']:
                st.markdown("**Additional Context:**")
                st.write(item['context'])
            st.markdown("**Quality Score:**")
            st.write(f"{item['quality_score']}%")

# Add download functionality in sidebar
if st.session_state.history:
    latest_req = st.session_state.history[-1]

    # Create a formatted string for the downloadable document
    download_str = f"""# Requirements Document - Iteration {st.session_state.iteration_count}

## Project Description
{latest_req['project_info']}

## Additional Context
{latest_req['context'] or 'No additional context provided.'}

## Current Requirements
{st.session_state.current_requirements}

## Latest Validation Feedback
{st.session_state.current_validation}

## Quality Metrics
- Overall Quality Score: {st.session_state.quality_score}%
- Ambiguities: {st.session_state.validation_metrics['ambiguities']}
- Inconsistencies: {st.session_state.validation_metrics['inconsistencies']}
- Missing Elements: {st.session_state.validation_metrics['missing_elements']}
- Conflicts: {st.session_state.validation_metrics['conflicts']}

## Requirements History
"""

    # Add complete history to the download document
    for item in st.session_state.history:
        download_str += f"\n### Iteration {item['iteration']}\n"
        download_str += f"\nRequirements:\n{item['requirements']}\n"
        download_str += f"\nValidation Feedback:\n{item['validation']}\n"
        download_str += f"\nQuality Score: {item['quality_score']}%\n"

    with st.sidebar:
        st.download_button(
            label="Download Requirements Document",
            data=download_str,
            file_name="requirements.md",
            mime="text/markdown",
            type="primary"
        )