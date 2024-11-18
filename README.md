# ByThePeople Bills Assistant - Setup and API Documentation

## Introduction

The **ByThePeople Bills Assistant** backend is a Flask-based API server that powers the AI-driven conversational IBM Watsonx assistant. It provides endpoints for searching, comparing, and analyzing Kenyan parliamentary bills, facilitating enhanced accessibility and understanding for both legislators and citizens.

---

## Backend Setup

### Prerequisites

- **Python 3.7 or higher**
- **pip** (Python package installer)
- **OpenAI API Key** (for embedding generation and GPT-4o-mini model usage)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/nairdaee/ibm-hackathon
   cd ibm-hackathon
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```


### Environment Variables

Set up the following environment variable:

- **`OPENAI_API_KEY`**: Your OpenAI API key for accessing embedding models and GPT-4o-mini.

You can set the environment variable in your `.env` file as follows:

```bash
OPENAI_API_KEY='your-openai-api-key'
```

### Running the Server

   ```bash
   python bills_api.py
   ```

   The server will run on `http://0.0.0.0:3000` by default.

---

## API Documentation

Postman: [IBM Hackathon Bills API](https://newage-ai.postman.co/workspace/NewAge-AI-Workspace~e396c6b2-9ada-4c2f-839d-148a41c1ba5a/collection/34830999-93f97a01-a029-4093-b023-54de3f4769b5?action=share&creator=34830999) 

## Conclusion

The ByThePeople Bills Assistant backend provides robust APIs for interacting with parliamentary bills, supporting advanced features like similarity search and bill comparison. By following this guide, you should be able to set up the backend server, understand the API endpoints, and integrate them with the frontend or IBM watsonx Assistant to deliver a powerful legislative tool.

For any issues or questions, please contact the team leader: [Adrian](mailto:etadriano2@gmail.com).

---

**Attribution:** All the bills data is publicly available on the `parliament.go.ke` website