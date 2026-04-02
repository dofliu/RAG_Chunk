# CLAUDE.md

## Project Overview

RAG_Chunk is an **Embedding & RAG Comparison Lab** — a React + TypeScript + Vite web app for comparing embedding models and RAG strategies side-by-side. It supports multi-format document upload, multiple chunking strategies, and parallel evaluation across different embedding/LLM model combinations.

## Tech Stack

- **Frontend**: React 19, TypeScript 5.8, Vite 6.2
- **Styling**: Tailwind CSS 4.1 (utility classes via `cn()` helper from clsx + tailwind-merge)
- **AI**: Google Gemini API (`@google/genai`) for embeddings and LLM generation
- **Storage**: IndexedDB via `idb-keyval` for embedding cache
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
  App.tsx              # Main React component (~976 lines, all UI + state logic)
  main.tsx             # Entry point
  index.css            # Tailwind config
  lib/
    rag.ts             # Core RAG logic: chunkText(), cosineSimilarity(), searchChunks()
    documentParser.ts  # Multi-format parser (PDF, DOCX, XLSX, TXT, MD, JSON)
    utils.ts           # cn() utility for Tailwind class merging
```

- **Single-component architecture**: `App.tsx` is the main monolithic component containing all state management and UI
- **Library functions** are separated into `src/lib/`
- No backend — this is a client-side-only app (API key is exposed to client, designed for AI Studio deployment)

## Environment

Requires a `.env` file (see `.env.example`):
```
GEMINI_API_KEY=<your-key>
```

The key is injected via Vite's `define` in `vite.config.ts` as `process.env.GEMINI_API_KEY`.

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

## Key Models & Constants

```typescript
AVAILABLE_MODELS = ['gemini-embedding-2-preview', 'gemini-embedding-001']
AVAILABLE_LLM_MODELS = ['gemini-3-flash-preview', 'gemini-3.1-pro-preview']
```

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
