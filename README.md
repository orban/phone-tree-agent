# AI Voice Agent Discovery Tool

This repository contains a tool for automatically discovering and mapping out the conversation flow of AI Voice Agents used by small businesses. The tool simulates phone calls to explore all possible paths in the agent's decision tree, creating a comprehensive map of the agent's capabilities.

## Purpose and Components

The main purpose of this tool is to automate the discovery process of AI Voice Agent scenarios. It consists of several key components working together:

1. **DiscoveryAgent**: Orchestrates the exploration process, making calls and building the phone tree.
2. **PhoneTree**: Represents the structure of the conversation flow.
3. **CallManager**: Handles the actual phone calls and transcription.
4. **OutputGenerator**: Generates various outputs like JSON, Mermaid graphs, and summary reports.

These components work together to explore the AI Voice Agent's decision tree, transcribe the conversations, extract the conversation flow, and generate a structured representation of the agent's capabilities.

## Quickstart Guide

### Prerequisites

This project uses `uv` for package management instead of `pip` or `poetry`. To install `uv`, run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/orban/hamming-takehome.git
   cd hamming-takehome
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   uv venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
   uv sync
   ```

3. Set up your environment variables in a `.env` file:

   ```bash
   OPENAI_API_KEY=your_openai_api_key
   AGENT_PHONE_NUMBER=+1234567890
   WEBHOOK_PORT=8000
   ```

See `.env.example` for required environment variables.

### Running the Program

To run the program, use:

```bash
uv run main.py
```

This will start the discovery process, make calls to the AI Voice Agent, and generate the output files.

## Example Output

Here's an example of a transcription and the resulting phone tree:

### Transcription

Agent: Hello, thank you for calling Anthem Air Conditioning and Plumbing. This is Olivia speaking. Are you an existing customer?

Customer: Yes, I am.

Agent: Great! Is this an emergency?

Customer: No, it's not an emergency.

Agent: Alright. What kind of issue are you facing?

Customer: I'm having a problem with my heating.

Agent: I understand. We can help with that. An agent will call you back shortly to schedule an appointment. Is there anything else I can help you with?

Customer: No, that's all. Thank you.

Agent: You're welcome. Have a great day! Goodbye.

Customer: Goodbye.

### Resulting Phone Tree

```python
Final Phone Tree Structure:
└── root
    └── existing_customer
        └── not_emergency
            └── not_emergency
                ├── heating_issue
                └── air_conditioning_issue             
```
