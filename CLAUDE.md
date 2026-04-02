# CLAUDE.md

## Project Overview

RAG_Chunk is an **Embedding & RAG Comparison Lab** — a React + TypeScript + Vite web app for comparing embedding models and RAG strategies side-by-side. It supports multi-format document upload, multiple chunking strategies, parallel evaluation across different embedding/LLM model combinations, and an **AutoResearch** engine that automatically finds the best configuration.

## Tech Stack

- **Frontend**: React 19, TypeScript 5.8, Vite 6.2
- **Styling**: Tailwind CSS 4.1 (utility classes via `cn()` helper from clsx + tailwind-merge)
- **AI**: Multi-provider embedding support (Google Gemini, OpenAI) + Gemini LLM generation
- **Storage**: IndexedDB via `idb-keyval` for embedding cache; localStorage for API keys
- **Document Parsing**: pdfjs-dist (PDF), mammoth (DOCX), xlsx (Excel), plain text
- **Export**: html2canvas + jsPDF for PDF report generation

## Commands

```bash
npm run dev        # Start dev server on http://localhost:3000
npm run build      # Production build to dist/
npm run preview    # Preview production build
npm run clean      # Remove dist/
npm run lint       # TypeScript type-check (tsc --noEmit)
```

There is no formal test framework (Jest/Vitest). Test files (`test_*.ts`) are manual scripts run with `tsx`.

## Project Structure

```
src/
  App.tsx              # Main React component (UI + state logic)
  main.tsx             # Entry point
  index.css            # Tailwind config
  lib/
    rag.ts             # Core RAG logic: chunkText(), cosineSimilarity(), searchChunks()
    documentParser.ts  # Multi-format parser (PDF, DOCX, XLSX, TXT, MD, JSON)
    utils.ts           # cn() utility for Tailwind class merging
    providers/
      types.ts         # EmbeddingProvider interface, ProviderModel type
      gemini.ts        # Google Gemini embedding provider
      openai.ts        # OpenAI embedding provider (text-embedding-3-small/large)
      registry.ts      # Provider registry: model lookup, embedWithModel(), API key persistence
    autoResearch/
      types.ts         # ExperimentConfig, SearchSpace, AutoResearchReport types
      scorer.ts        # Composite score calculation (faithfulness + relevance + latency)
      engine.ts        # AutoResearch engine: generates combos, runs experiments, collects results
```

### Architecture

- **Single-component UI**: `App.tsx` contains all state management and UI with two tabs: Search and AutoResearch
- **Provider abstraction**: `lib/providers/` decouples embedding logic from the UI. Each provider implements `EmbeddingProvider` interface. Adding a new provider = one file + register in `registry.ts`
- **AutoResearch engine**: `lib/autoResearch/engine.ts` takes a search space, generates all parameter combinations, runs the full RAG pipeline for each, evaluates quality, and returns ranked results
- No backend — this is a client-side-only app

## Environment

Optional `.env` file (see `.env.example`):
```
GEMINI_API_KEY=<your-key>
```

The Gemini key is injected via Vite's `define` in `vite.config.ts` as `process.env.GEMINI_API_KEY`. It seeds the initial API key in localStorage but can be overridden in the UI's API Keys panel.

API keys for all providers are managed in the UI and persisted to localStorage under `rag_lab_api_keys`.

## Code Conventions

- **Language**: TypeScript with React JSX
- **Naming**: camelCase for variables/functions, PascalCase for interfaces/types, UPPERCASE for constants
- **State management**: React `useState` hooks (no external state library)
- **Async**: async/await with try-catch error handling
- **Styling**: Inline Tailwind classes; use `cn()` for conditional class merging
- **Dark mode**: Toggled via `darkMode` boolean state, applied with conditional classes
- **Icons**: Lucide React
- **No strict TypeScript**: `strict` is not enabled in tsconfig; `experimentalDecorators` is on
- **Path alias**: `@/*` maps to root directory

## Embedding Providers

Providers are registered in `src/lib/providers/registry.ts`:

| Provider | Models | Notes |
|----------|--------|-------|
| **Gemini** | `gemini-embedding-2-preview`, `gemini-embedding-001` | Uses `@google/genai` SDK |
| **OpenAI** | `text-embedding-3-small`, `text-embedding-3-large` | Direct REST API calls |

To add a new provider: create `src/lib/providers/<name>.ts` implementing `EmbeddingProvider`, then add it to the `providers` array in `registry.ts`.

## Key Constants

```typescript
AVAILABLE_LLM_MODELS = ['gemini-3-flash-preview', 'gemini-3.1-pro-preview']
```

Embedding models are dynamically loaded from the provider registry.

## Caching

Embeddings are cached in IndexedDB with key format:
```
embeddings_${filename}_${filesize}_${chunkSize}_${overlap}
```

## Important Notes

- Colors use hex values (not oklch) for html2canvas PDF export compatibility
- Rate limiting: batch processing is implemented for API calls
- The app is frontend-only — do not introduce backend dependencies for core features
- README.md is written in Traditional Chinese
