import { useState, ChangeEvent } from 'react';
import { Upload, FileText, Settings, Search, Loader2 } from 'lucide-react';
import { parseDocument } from './lib/documentParser';
import { chunkText, searchChunks, ChunkWithScore } from './lib/rag';
import { GoogleGenAI } from '@google/genai';
import { cn } from './lib/utils';
import Markdown from 'react-markdown';
import { get, set, clear } from 'idb-keyval';

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

const AVAILABLE_MODELS = [
  'gemini-embedding-2-preview',
  'gemini-embedding-001'
];

interface ModelResult {
  retrievedChunks: ChunkWithScore[];
  answer: string;
}

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [documentText, setDocumentText] = useState('');
  const [chunkSize, setChunkSize] = useState(1000);
  const [overlap, setOverlap] = useState(200);
  const [selectedModels, setSelectedModels] = useState<string[]>(['gemini-embedding-2-preview', 'gemini-embedding-001']);
  
  const [isProcessing, setIsProcessing] = useState(false);
  const [processStatus, setProcessStatus] = useState('');
  const [embeddings, setEmbeddings] = useState<Record<string, {text: string, embedding: number[]}[]>>({});
  
  const [query, setQuery] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [results, setResults] = useState<Record<string, ModelResult>>({});
  const [errorMsg, setErrorMsg] = useState('');

  const handleFileChange = async (e: ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;
    setFile(selectedFile);
    setDocumentText('');
    setEmbeddings({});
    setResults({});
  };

  const handleProcess = async () => {
    setErrorMsg('');
    if (!file) return;
    if (selectedModels.length === 0) {
      setErrorMsg('Please select at least one model.');
      return;
    }

    setIsProcessing(true);
    setProcessStatus('Checking cache...');
    try {
      const cacheKey = `embeddings_${file.name}_${file.size}_${chunkSize}_${overlap}`;
      const cachedEmbeddings: Record<string, {text: string, embedding: number[]}[]> = await get(cacheKey) || {};
      
      const modelsToProcess = selectedModels.filter(m => !cachedEmbeddings[m]);
      
      if (modelsToProcess.length === 0) {
        setEmbeddings(cachedEmbeddings);
        setProcessStatus('Loaded from cache!');
        setIsProcessing(false);
        return;
      }

      setProcessStatus('Parsing document...');
      let text = documentText;
      if (!text) {
        text = await parseDocument(file);
        setDocumentText(text);
      }

      setProcessStatus('Chunking text...');
      const chunks = chunkText(text, chunkSize, overlap);
      
      if (chunks.length === 0) {
        throw new Error('No text chunks generated.');
      }

      const newEmbeddings: Record<string, {text: string, embedding: number[]}[]> = { ...cachedEmbeddings };

      for (const model of modelsToProcess) {
        setProcessStatus(`Generating embeddings with ${model}...`);
        const modelEmbeddings: {text: string, embedding: number[]}[] = [];
        
        // Batch processing to avoid payload limits
        const batchSize = 10;
        for (let i = 0; i < chunks.length; i += batchSize) {
          const batch = chunks.slice(i, i + batchSize);
          const result = await ai.models.embedContent({
            model: model,
            contents: batch,
          });
          
          if (result.embeddings) {
            result.embeddings.forEach((emb: any, idx: number) => {
              modelEmbeddings.push({
                text: batch[idx],
                embedding: emb.values
              });
            });
          }
        }
        newEmbeddings[model] = modelEmbeddings;
      }

      await set(cacheKey, newEmbeddings);
      setEmbeddings(newEmbeddings);
      setProcessStatus('Processing complete!');
    } catch (error: any) {
      console.error(error);
      setErrorMsg(`Error processing document: ${error.message}`);
      setProcessStatus('Error occurred.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleSearch = async () => {
    setErrorMsg('');
    if (!query.trim()) return;
    if (Object.keys(embeddings).length === 0) {
      setErrorMsg('Please process a document first.');
      return;
    }

    setIsSearching(true);
    const newResults: Record<string, ModelResult> = {};

    try {
      for (const model of selectedModels) {
        if (!embeddings[model]) continue;

        // Embed query
        const queryResult = await ai.models.embedContent({
          model: model,
          contents: query,
        });
        
        const queryEmbedding = queryResult.embeddings?.[0]?.values;
        if (!queryEmbedding) continue;

        // Search chunks
        const retrievedChunks = searchChunks(queryEmbedding, embeddings[model], 3);

        // Generate answer
        const context = retrievedChunks.map((c, i) => `[Chunk ${i + 1}]:\n${c.text}`).join('\n\n---\n\n');
        const prompt = `You are a helpful assistant. Answer the user's question based ONLY on the provided context. If the answer is not in the context, say "I cannot answer this based on the provided document."\n\nWhen providing your answer, you MUST cite the source chunks you used by referencing their numbers like [Chunk 1] or [Chunk 2, 3].\n\nContext:\n${context}\n\nQuestion: ${query}`;
        
        const answerResult = await ai.models.generateContent({
          model: 'gemini-3-flash-preview',
          contents: prompt,
        });

        newResults[model] = {
          retrievedChunks,
          answer: answerResult.text || 'No answer generated.'
        };
      }

      setResults(newResults);
    } catch (error: any) {
      console.error(error);
      setErrorMsg(`Error during search: ${error.message}`);
    } finally {
      setIsSearching(false);
    }
  };

  const handleClearCache = async () => {
    try {
      await clear();
      setEmbeddings({});
      setResults({});
      setProcessStatus('Cache cleared!');
      setTimeout(() => setProcessStatus(''), 3000);
    } catch (err) {
      console.error(err);
      setErrorMsg('Failed to clear cache.');
    }
  };

  return (
    <div className="min-h-screen bg-zinc-50 text-zinc-900 font-sans flex flex-col md:flex-row">
      {/* Sidebar */}
      <div className="w-full md:w-80 bg-white border-r border-zinc-200 p-6 flex flex-col gap-6 overflow-y-auto shrink-0">
        <div>
          <h1 className="text-xl font-semibold tracking-tight mb-1">Embedding Compare</h1>
          <p className="text-sm text-zinc-500">Test and compare Gemini embedding models.</p>
        </div>

        <div className="space-y-4">
          <h2 className="text-sm font-medium flex items-center gap-2">
            <Upload className="w-4 h-4" /> Document Upload
          </h2>
          <div className="border-2 border-dashed border-zinc-200 rounded-xl p-4 text-center hover:bg-zinc-50 transition-colors">
            <input
              type="file"
              id="file-upload"
              className="hidden"
              accept=".txt,.pdf,.docx,.xlsx,.xls"
              onChange={handleFileChange}
            />
            <label htmlFor="file-upload" className="cursor-pointer flex flex-col items-center gap-2">
              <FileText className="w-8 h-8 text-zinc-400" />
              <span className="text-sm font-medium text-zinc-700">
                {file ? file.name : 'Click to upload'}
              </span>
              <span className="text-xs text-zinc-500">PDF, DOCX, EXCEL, TXT</span>
            </label>
          </div>
        </div>

        <div className="space-y-4">
          <h2 className="text-sm font-medium flex items-center gap-2">
            <Settings className="w-4 h-4" /> Chunking Strategy
          </h2>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-zinc-500 mb-1">Chunk Size</label>
              <input
                type="number"
                value={chunkSize}
                onChange={(e) => setChunkSize(Number(e.target.value))}
                className="w-full px-3 py-2 bg-zinc-50 border border-zinc-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500"
              />
            </div>
            <div>
              <label className="block text-xs text-zinc-500 mb-1">Overlap</label>
              <input
                type="number"
                value={overlap}
                onChange={(e) => setOverlap(Number(e.target.value))}
                className="w-full px-3 py-2 bg-zinc-50 border border-zinc-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500"
              />
            </div>
          </div>
        </div>

        <div className="space-y-3">
          <h2 className="text-sm font-medium">Models to Compare</h2>
          <div className="space-y-2">
            {AVAILABLE_MODELS.map(model => (
              <label key={model} className="flex items-center gap-3 p-3 border border-zinc-200 rounded-lg cursor-pointer hover:bg-zinc-50">
                <input
                  type="checkbox"
                  checked={selectedModels.includes(model)}
                  onChange={(e) => {
                    if (e.target.checked) {
                      setSelectedModels([...selectedModels, model]);
                    } else {
                      setSelectedModels(selectedModels.filter(m => m !== model));
                    }
                  }}
                  className="w-4 h-4 text-indigo-600 rounded border-zinc-300 focus:ring-indigo-500"
                />
                <span className="text-sm font-medium">{model}</span>
              </label>
            ))}
          </div>
        </div>

        {errorMsg && (
          <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-600">
            {errorMsg}
          </div>
        )}

        <div className="space-y-2">
          <button
            onClick={handleProcess}
            disabled={!file || isProcessing || selectedModels.length === 0}
            className="w-full py-2.5 bg-zinc-900 text-white rounded-lg text-sm font-medium hover:bg-zinc-800 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-colors"
          >
            {isProcessing ? (
              <><Loader2 className="w-4 h-4 animate-spin" /> {processStatus}</>
            ) : (
              'Process Document'
            )}
          </button>
          
          <button
            onClick={handleClearCache}
            disabled={isProcessing}
            className="w-full py-2 bg-white border border-zinc-200 text-zinc-600 rounded-lg text-sm font-medium hover:bg-zinc-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Clear Saved Embeddings
          </button>
          
          {processStatus && !isProcessing && processStatus !== 'Error occurred.' && (
            <p className="text-xs text-center text-emerald-600 font-medium">{processStatus}</p>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col h-screen overflow-hidden">
        <div className="p-6 border-b border-zinc-200 bg-white">
          <div className="max-w-3xl mx-auto">
            <div className="relative">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                placeholder="Ask a question about your document..."
                className="w-full pl-12 pr-24 py-4 bg-zinc-50 border border-zinc-200 rounded-2xl text-base focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 shadow-sm"
              />
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-zinc-400" />
              <button
                onClick={handleSearch}
                disabled={isSearching || !query.trim() || Object.keys(embeddings).length === 0}
                className="absolute right-2 top-1/2 -translate-y-1/2 px-4 py-2 bg-indigo-600 text-white rounded-xl text-sm font-medium hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 transition-colors"
              >
                {isSearching ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Search'}
              </button>
            </div>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-6">
          <div className="max-w-7xl mx-auto">
            {Object.keys(results).length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {selectedModels.map(model => results[model] && (
                  <div key={model} className="space-y-6">
                    <div className="flex items-center justify-between border-b border-zinc-200 pb-4">
                      <h2 className="text-xl font-semibold text-zinc-900">{model}</h2>
                    </div>

                    <div className="space-y-4">
                      <h3 className="text-sm font-medium text-zinc-500 uppercase tracking-wider">Generated Answer</h3>
                      <div className="bg-white p-6 rounded-2xl border border-zinc-200 shadow-sm min-h-[150px]">
                        <div className="text-zinc-700 leading-relaxed prose prose-sm max-w-none prose-zinc">
                          <Markdown>{results[model].answer}</Markdown>
                        </div>
                      </div>
                    </div>

                    <div className="space-y-4">
                      <h3 className="text-sm font-medium text-zinc-500 uppercase tracking-wider">Retrieved Context (Top 3)</h3>
                      <div className="space-y-3">
                        {results[model].retrievedChunks.map((chunk, i) => (
                          <div key={i} className="bg-white p-4 rounded-xl border border-zinc-200 shadow-sm">
                            <div className="flex items-center justify-between mb-2">
                              <span className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Chunk {i + 1}</span>
                              <span className="text-xs font-mono text-indigo-600 bg-indigo-50 px-2 py-1 rounded-md">
                                Score: {chunk.score.toFixed(4)}
                              </span>
                            </div>
                            <p className="text-sm text-zinc-600 line-clamp-4">{chunk.text}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-zinc-400 space-y-4 py-20">
                <div className="w-16 h-16 bg-zinc-100 rounded-2xl flex items-center justify-center">
                  <Search className="w-8 h-8" />
                </div>
                <p className="text-sm">Upload a document and ask a question to see results.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
