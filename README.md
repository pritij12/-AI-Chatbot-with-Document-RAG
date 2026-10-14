# Chat with your PDF (simple RAG chatbot)

Upload a PDF, ask a question, and get an answer with sources.

## Files

```
simple-rag/
├── server.js        All the backend code (read PDF, chunk, embed, search, ask Claude)
├── public/
│   └── index.html   The web page (upload box, question box, answer)
├── package.json     List of packages
├── .env.example     Where your API keys go
└── README.md
```

## Run it (5 steps)

1. Install Node.js (version 18 or newer) from nodejs.org.
2. Open a terminal in this folder and run:
   ```
   npm install
   ```
3. Copy `.env.example` to a new file called `.env` and paste your two keys:
   - OPENAI_API_KEY (for embeddings)
   - ANTHROPIC_API_KEY (for Claude)
4. Start the app:
   ```
   npm start
   ```
5. Open http://localhost:3000, upload a PDF, and ask a question.

## How it works (the 7 steps)

1. You upload a PDF.
2. `pdf-parse` reads the text.
3. The text is split into chunks (about 500 tokens, 50 overlap).
4. Each chunk is turned into numbers (embeddings) using OpenAI `text-embedding-3-small`.
5. The chunks and numbers are saved in `store.json` (a simple file instead of a database).
6. Your question is also turned into numbers, and cosine similarity finds the 5 closest chunks.
7. Claude reads those 5 chunks and writes the answer, streamed live, with source numbers like [1].

## Good to know

- This simple version saves data in `store.json`. A real project would use a database with pgvector.
- Scanned PDFs (pictures of text) do not work.
- Every upload and question uses your OpenAI and Anthropic credits. Never share your `.env` file.
