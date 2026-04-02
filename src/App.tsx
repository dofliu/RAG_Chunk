import { useState, ChangeEvent, useEffect } from 'react';
import { Upload, FileText, Settings, Search, Loader2, Trash2, Moon, Sun, Database, MessageSquare } from 'lucide-react';
import { parseDocument } from './lib/documentParser';
import { chunkText, searchChunks, ChunkWithScore, cosineSimilarity } from './lib/rag';
import { GoogleGenAI } from '@google/genai';
import { cn } from './lib/utils';
import Markdown from 'react-markdown';
import { get, set, clear, keys, del } from 'idb-keyval';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

const AVAILABLE_MODELS = [
  'gemini-embedding-2-preview',
  'gemini-embedding-001'
];

const AVAILABLE_LLM_MODELS = [
  'gemini-3-flash-preview',
  'gemini-3.1-pro-preview'
];

interface ModelResult {
  retrievedChunks: ChunkWithScore[];
  reRankedChunks?: ChunkWithScore[];
  answers: Record<string, Record<number, { 
    text: string, 
    usage?: any,
    evaluation?: { faithfulness: number, relevance: number, reasoning: string },
    history: { role: 'user' | 'model', parts: { text: string }[] }[] 
  }>>;
}

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [documentText, setDocumentText] = useState('');
  const [chunkSize, setChunkSize] = useState(1000);
  const [overlap, setOverlap] = useState(200);
  const [chunkingStrategy, setChunkingStrategy] = useState<'fixed' | 'paragraph' | 'semantic'>('fixed');
  const [kValue, setKValue] = useState(3); 
  const [selectedKValues, setSelectedKValues] = useState<number[]>([3, 5]);
  const [selectedModels, setSelectedModels] = useState<string[]>(['gemini-embedding-2-preview']);
  const [selectedLLMModels, setSelectedLLMModels] = useState<string[]>(['gemini-3-flash-preview']);
  const [systemPrompt, setSystemPrompt] = useState('You are a helpful assistant. Answer the user\'s question based ONLY on the provided context. If the answer is not in the context, say "I cannot answer this based on the provided document."\n\nWhen providing your answer, you MUST cite the source chunks you used by referencing their numbers like [Chunk 1] or [Chunk 2, 3].');
  const [threshold, setThreshold] = useState(0.3);
  const [useReRanking, setUseReRanking] = useState(false);
  const [useEvaluation, setUseEvaluation] = useState(false);
  
  const [isProcessing, setIsProcessing] = useState(false);
  const [processStatus, setProcessStatus] = useState('');
  const [embeddings, setEmbeddings] = useState<Record<string, {text: string, embedding: number[]}[]>>({});
  const [cachedFiles, setCachedFiles] = useState<string[]>([]);
  const [darkMode, setDarkMode] = useState(false);
  
  const [query, setQuery] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [results, setResults] = useState<Record<string, ModelResult>>({});
  const [errorMsg, setErrorMsg] = useState('');

  useEffect(() => {
    refreshCachedFiles();
    // Check system dark mode
    if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
      setDarkMode(true);
    }
  }, []);

  const refreshCachedFiles = async () => {
    const allKeys = await keys();
    const fileKeys = allKeys.filter(k => typeof k === 'string' && k.startsWith('embeddings_')) as string[];
    setCachedFiles(fileKeys);
  };

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
      let chunks: string[] = [];
      
      if (chunkingStrategy === 'paragraph') {
        chunks = text.split(/\n\s*\n/).filter(p => p.trim().length > 0);
      } else if (chunkingStrategy === 'semantic') {
        setProcessStatus('Semantic chunking (embedding sentences)...');
        // Split into sentences (basic regex)
        const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
        if (sentences.length <= 1) {
          chunks = [text];
        } else {
          // Embed sentences in batches to avoid rate limits
          const sentenceEmbeddings: number[][] = [];
          const batchSize = 20;
          for (let i = 0; i < sentences.length; i += batchSize) {
            const batch = sentences.slice(i, i + batchSize);
            const result = await ai.models.embedContent({
              model: 'models/text-embedding-004',
              contents: batch,
            });
            sentenceEmbeddings.push(...result.embeddings.map(e => e.values));
          }

          // Merge sentences based on similarity
          let currentChunk = sentences[0];
          for (let i = 0; i < sentenceEmbeddings.length - 1; i++) {
            const sim = cosineSimilarity(sentenceEmbeddings[i], sentenceEmbeddings[i+1]);
            // If similarity is high, merge. Threshold 0.7 is arbitrary.
            if (sim > 0.7 && currentChunk.length < chunkSize) {
              currentChunk += " " + sentences[i+1];
            } else {
              chunks.push(currentChunk);
              currentChunk = sentences[i+1];
            }
          }
          chunks.push(currentChunk);
        }
      } else {
        // Fixed size
        chunks = chunkText(text, chunkSize, overlap);
      }
      
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
      refreshCachedFiles();
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

        // Search chunks (always get 10 for display)
        const allRetrievedChunks = searchChunks(queryEmbedding, embeddings[model], 10);
        
        let finalChunks = allRetrievedChunks;

        // Re-ranking step
        if (useReRanking) {
          setProcessStatus(`Re-ranking chunks for ${model}...`);
          const reRankPrompt = `You are a search expert. Rank the following text chunks by their relevance to the query: "${query}".
          Return ONLY a JSON array of the indices (0-9) in order of relevance.
          Example: [3, 0, 5, 1, 2, 4, 6, 7, 8, 9]
          
          Chunks:
          ${allRetrievedChunks.map((c, i) => `[${i}]: ${c.text.substring(0, 300)}...`).join('\n')}`;
          
          try {
            const reRankResult = await ai.models.generateContent({
              model: 'gemini-3-flash-preview',
              contents: reRankPrompt,
              config: { responseMimeType: 'application/json' }
            });
            const newIndices = JSON.parse(reRankResult.text || '[]');
            if (Array.isArray(newIndices)) {
              finalChunks = newIndices
                .map(idx => allRetrievedChunks[idx])
                .filter(c => c !== undefined);
            }
          } catch (e) {
            console.error('Re-ranking failed:', e);
          }
        }

        // Filter by threshold
        const retrievedChunks = finalChunks.filter(c => c.score >= threshold);
        
        const modelAnswers: Record<string, Record<number, { text: string, usage?: any, history: any[], evaluation?: any }>> = {};

        // Generate answers for each selected LLM model
        for (const llmModel of selectedLLMModels) {
          modelAnswers[llmModel] = {};
          
          // Generate answers for each selected K value
          for (const k of selectedKValues) {
            const chunksForLLM = retrievedChunks.slice(0, k);
            if (chunksForLLM.length === 0) {
              modelAnswers[llmModel][k] = { text: 'No chunks met the similarity threshold.', history: [] };
              continue;
            }

            const context = chunksForLLM.map((c, i) => `[Chunk ${i + 1}]:\n${c.text}`).join('\n\n---\n\n');
            
            // Get previous history for this specific model/K
            const prevHistory = results[model]?.answers[llmModel]?.[k]?.history || [];
            
            const currentPrompt = `${systemPrompt}\n\nContext:\n${context}\n\nQuestion: ${query}`;
            
            const newHistory = [
              ...prevHistory,
              { role: 'user', parts: [{ text: currentPrompt }] }
            ];

            const answerResult = await ai.models.generateContent({
              model: llmModel,
              contents: newHistory,
            });
            
            const answerText = answerResult.text || 'No answer generated.';
            
            let evaluation = undefined;
            if (useEvaluation) {
              setProcessStatus(`Evaluating answer (${llmModel}, K=${k})...`);
              const evalPrompt = `Evaluate the following RAG answer based on the provided context and query.
              Metrics:
              1. Faithfulness (0-10): Is the answer grounded ONLY in the context?
              2. Relevance (0-10): Does the answer directly address the query?
              
              Return ONLY a JSON object: { "faithfulness": number, "relevance": number, "reasoning": "string" }
              
              Query: ${query}
              Context: ${context}
              Answer: ${answerText}`;
              
              try {
                const evalResult = await ai.models.generateContent({
                  model: 'gemini-3-flash-preview',
                  contents: evalPrompt,
                  config: { responseMimeType: 'application/json' }
                });
                evaluation = JSON.parse(evalResult.text || '{}');
              } catch (e) {
                console.error('Evaluation failed:', e);
              }
            }

            modelAnswers[llmModel][k] = { 
              text: answerText,
              usage: answerResult.usageMetadata,
              evaluation,
              history: [
                ...newHistory,
                { role: 'model', parts: [{ text: answerText }] }
              ]
            };
          }
        }

        newResults[model] = {
          retrievedChunks: allRetrievedChunks,
          reRankedChunks: useReRanking ? finalChunks : undefined,
          answers: modelAnswers
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
      setCachedFiles([]);
      setProcessStatus('All cache cleared!');
      setTimeout(() => setProcessStatus(''), 3000);
    } catch (err) {
      console.error(err);
      setErrorMsg('Failed to clear cache.');
    }
  };

  const handleDeleteCacheItem = async (key: string) => {
    try {
      await del(key);
      refreshCachedFiles();
      setProcessStatus('Item deleted!');
      setTimeout(() => setProcessStatus(''), 3000);
    } catch (err) {
      console.error(err);
      setErrorMsg('Failed to delete item.');
    }
  };

  const handleExportPDF = async () => {
    const element = document.getElementById('results-container');
    if (!element) return;

    setIsSearching(true); // Reuse searching state for loading
    setProcessStatus('Generating PDF...');

    try {
      const canvas = await html2canvas(element, {
        scale: 2,
        useCORS: true,
        logging: false,
        backgroundColor: '#f9fafb'
      });
      
      const imgData = canvas.toDataURL('image/png');
      const pdf = new jsPDF('p', 'mm', 'a4');
      const imgProps = pdf.getImageProperties(imgData);
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = (imgProps.height * pdfWidth) / imgProps.width;
      
      // Handle multi-page if needed
      let heightLeft = pdfHeight;
      let position = 0;
      const pageHeight = pdf.internal.pageSize.getHeight();

      pdf.addImage(imgData, 'PNG', 0, position, pdfWidth, pdfHeight);
      heightLeft -= pageHeight;

      while (heightLeft >= 0) {
        position = heightLeft - pdfHeight;
        pdf.addPage();
        pdf.addImage(imgData, 'PNG', 0, position, pdfWidth, pdfHeight);
        heightLeft -= pageHeight;
      }

      pdf.save(`RAG_Compare_Report_${new Date().getTime()}.pdf`);
      setProcessStatus('PDF Downloaded!');
      setTimeout(() => setProcessStatus(''), 3000);
    } catch (error) {
      console.error('PDF generation failed:', error);
      setErrorMsg('Failed to generate PDF. Please try again.');
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <div className={cn(
      "min-h-screen font-sans flex flex-col md:flex-row transition-colors duration-300",
      darkMode ? "bg-zinc-950 text-zinc-100" : "bg-zinc-50 text-zinc-900"
    )}>
      {/* Sidebar */}
      <div className={cn(
        "w-full md:w-80 border-r p-6 flex flex-col gap-6 overflow-y-auto shrink-0 transition-colors duration-300",
        darkMode ? "bg-zinc-900 border-zinc-800" : "bg-white border-zinc-200"
      )}>
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold tracking-tight mb-1">RAG Lab</h1>
            <p className={cn("text-xs", darkMode ? "text-zinc-400" : "text-zinc-500")}>Compare models & K-values.</p>
          </div>
          <button 
            onClick={() => setDarkMode(!darkMode)}
            className={cn(
              "p-2 rounded-lg transition-colors",
              darkMode ? "bg-zinc-800 text-yellow-400 hover:bg-zinc-700" : "bg-zinc-100 text-zinc-600 hover:bg-zinc-200"
            )}
          >
            {darkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>
        </div>

        <div className="space-y-4">
          <h2 className="text-sm font-medium flex items-center gap-2">
            <Upload className="w-4 h-4" /> Document Upload
          </h2>
          <div className={cn(
            "border-2 border-dashed rounded-xl p-4 text-center transition-colors",
            darkMode ? "border-zinc-800 hover:bg-zinc-800/50" : "border-zinc-200 hover:bg-zinc-50"
          )}>
            <input
              type="file"
              id="file-upload"
              className="hidden"
              accept=".txt,.pdf,.docx,.xlsx,.xls,.md,.json"
              onChange={handleFileChange}
            />
            <label htmlFor="file-upload" className="cursor-pointer flex flex-col items-center gap-2">
              <FileText className={cn("w-8 h-8", darkMode ? "text-zinc-700" : "text-zinc-400")} />
              <span className={cn("text-sm font-medium", darkMode ? "text-zinc-300" : "text-zinc-700")}>
                {file ? file.name : 'Click to upload'}
              </span>
              <span className="text-[10px] text-zinc-500 uppercase tracking-widest">PDF, DOCX, XLS, TXT, MD, JSON</span>
            </label>
          </div>
        </div>

        <div className="space-y-4">
          <h2 className="text-sm font-medium flex items-center gap-2">
            <Settings className="w-4 h-4" /> RAG Strategy
          </h2>
          <div className="space-y-4">
            <div className="space-y-2">
              <label className="block text-[10px] text-zinc-500 uppercase font-bold mb-1">Chunking Strategy</label>
              <select
                value={chunkingStrategy}
                onChange={(e) => setChunkingStrategy(e.target.value as any)}
                className={cn(
                  "w-full px-3 py-2 border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500",
                  darkMode ? "bg-zinc-800 border-zinc-700 text-zinc-100" : "bg-zinc-50 border-zinc-200 text-zinc-900"
                )}
              >
                <option value="fixed">Fixed Size</option>
                <option value="paragraph">Paragraph</option>
                <option value="semantic">Semantic (Experimental)</option>
              </select>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-[10px] text-zinc-500 uppercase font-bold mb-1">Chunk Size</label>
                <input
                  type="number"
                  value={chunkSize}
                  onChange={(e) => setChunkSize(Number(e.target.value))}
                  className={cn(
                    "w-full px-3 py-2 border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500",
                    darkMode ? "bg-zinc-800 border-zinc-700 text-zinc-100" : "bg-zinc-50 border-zinc-200 text-zinc-900"
                  )}
                />
              </div>
              <div>
                <label className="block text-[10px] text-zinc-500 uppercase font-bold mb-1">Overlap</label>
                <input
                  type="number"
                  value={overlap}
                  disabled={chunkingStrategy !== 'fixed'}
                  onChange={(e) => setOverlap(Number(e.target.value))}
                  className={cn(
                    "w-full px-3 py-2 border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500",
                    darkMode ? "bg-zinc-800 border-zinc-700 text-zinc-100" : "bg-zinc-50 border-zinc-200 text-zinc-900",
                    chunkingStrategy !== 'fixed' && "opacity-50 cursor-not-allowed"
                  )}
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between items-center mb-1">
                <label className="block text-[10px] text-zinc-500 uppercase font-bold">Similarity Threshold</label>
                <span className="text-xs font-mono text-indigo-500 font-bold">{threshold.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
                className="w-full h-1.5 bg-zinc-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
              />
            </div>

            <div>
              <label className="block text-[10px] text-zinc-500 uppercase font-bold mb-2">Compare K Values</label>
              <div className="flex gap-2">
                {[3, 5, 7, 10].map((v) => (
                  <button
                    key={v}
                    onClick={() => {
                      if (selectedKValues.includes(v)) {
                        if (selectedKValues.length > 1) {
                          setSelectedKValues(selectedKValues.filter(k => k !== v));
                        }
                      } else {
                        setSelectedKValues([...selectedKValues, v].sort((a, b) => a - b));
                      }
                    }}
                    className={cn(
                      "flex-1 py-1.5 text-xs font-medium rounded-md border transition-all",
                      selectedKValues.includes(v)
                        ? "bg-indigo-600 border-indigo-600 text-white shadow-sm"
                        : darkMode ? "bg-zinc-800 border-zinc-700 text-zinc-400 hover:border-zinc-600" : "bg-white border-zinc-200 text-zinc-600 hover:border-zinc-300"
                    )}
                  >
                    {v}
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-2 pt-2">
              <label className="flex items-center gap-3 p-2 border border-zinc-200 rounded-lg cursor-pointer hover:bg-zinc-50 transition-colors">
                <input
                  type="checkbox"
                  checked={useReRanking}
                  onChange={(e) => setUseReRanking(e.target.checked)}
                  className="w-4 h-4 text-indigo-600 rounded border-zinc-300 focus:ring-indigo-500"
                />
                <div className="flex flex-col">
                  <span className="text-xs font-bold">Enable Re-ranking</span>
                  <span className="text-[9px] text-zinc-500">Use LLM to refine retrieval order.</span>
                </div>
              </label>
              <label className="flex items-center gap-3 p-2 border border-zinc-200 rounded-lg cursor-pointer hover:bg-zinc-50 transition-colors">
                <input
                  type="checkbox"
                  checked={useEvaluation}
                  onChange={(e) => setUseEvaluation(e.target.checked)}
                  className="w-4 h-4 text-indigo-600 rounded border-zinc-300 focus:ring-indigo-500"
                />
                <div className="flex flex-col">
                  <span className="text-xs font-bold">Self-Evaluation</span>
                  <span className="text-[9px] text-zinc-500">Score answers on Faithfulness & Relevance.</span>
                </div>
              </label>
            </div>
          </div>
        </div>

        <div className="space-y-3">
          <h2 className="text-sm font-medium flex items-center gap-2">
            <Database className="w-4 h-4" /> Embedding Models
          </h2>
          <div className="space-y-2">
            {AVAILABLE_MODELS.map(model => (
              <label key={model} className={cn(
                "flex items-center gap-3 p-3 border rounded-lg cursor-pointer transition-colors",
                darkMode ? "border-zinc-800 hover:bg-zinc-800" : "border-zinc-200 hover:bg-zinc-50"
              )}>
                <input
                  type="checkbox"
                  checked={selectedModels.includes(model)}
                  onChange={(e) => {
                    if (e.target.checked) {
                      setSelectedModels([...selectedModels, model]);
                    } else {
                      if (selectedModels.length > 1) {
                        setSelectedModels(selectedModels.filter(m => m !== model));
                      }
                    }
                  }}
                  className="w-4 h-4 text-indigo-600 rounded border-zinc-300 focus:ring-indigo-500"
                />
                <span className="text-xs font-medium">{model}</span>
              </label>
            ))}
          </div>
        </div>

        <div className="space-y-3">
          <h2 className="text-sm font-medium flex items-center gap-2">
            <MessageSquare className="w-4 h-4" /> LLM Models
          </h2>
          <div className="space-y-2">
            {AVAILABLE_LLM_MODELS.map(model => (
              <label key={model} className={cn(
                "flex items-center gap-3 p-3 border rounded-lg cursor-pointer transition-colors",
                darkMode ? "border-zinc-800 hover:bg-zinc-800" : "border-zinc-200 hover:bg-zinc-50"
              )}>
                <input
                  type="checkbox"
                  checked={selectedLLMModels.includes(model)}
                  onChange={(e) => {
                    if (e.target.checked) {
                      setSelectedLLMModels([...selectedLLMModels, model]);
                    } else {
                      if (selectedLLMModels.length > 1) {
                        setSelectedLLMModels(selectedLLMModels.filter(m => m !== model));
                      }
                    }
                  }}
                  className="w-4 h-4 text-indigo-600 rounded border-zinc-300 focus:ring-indigo-500"
                />
                <span className="text-xs font-medium">{model}</span>
              </label>
            ))}
          </div>
        </div>

        <div className="space-y-3">
          <h2 className="text-sm font-medium">System Prompt</h2>
          <textarea
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            className={cn(
              "w-full h-32 px-3 py-2 border rounded-lg text-xs focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 resize-none",
              darkMode ? "bg-zinc-800 border-zinc-700 text-zinc-300" : "bg-zinc-50 border-zinc-200 text-zinc-600"
            )}
            placeholder="Enter custom RAG instructions..."
          />
        </div>

        <div className="space-y-2 mt-auto pt-4 border-t border-zinc-200 dark:border-zinc-800">
          <button
            onClick={handleProcess}
            disabled={!file || isProcessing || selectedModels.length === 0}
            className="w-full py-2.5 bg-indigo-600 text-white rounded-lg text-sm font-medium hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-colors"
          >
            {isProcessing ? (
              <><Loader2 className="w-4 h-4 animate-spin" /> {processStatus}</>
            ) : (
              'Process Document'
            )}
          </button>
          
          <div className="space-y-2">
            <h3 className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Local Cache</h3>
            <div className="max-h-32 overflow-y-auto space-y-1">
              {cachedFiles.map(key => {
                const fileName = key.replace('embeddings_', '').split('_')[0];
                return (
                  <div key={key} className={cn(
                    "flex items-center justify-between p-2 rounded text-[10px] group",
                    darkMode ? "bg-zinc-800/50 hover:bg-zinc-800" : "bg-zinc-100 hover:bg-zinc-200"
                  )}>
                    <span className="truncate flex-1 mr-2">{fileName}</span>
                    <button 
                      onClick={() => handleDeleteCacheItem(key)}
                      className="text-zinc-400 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-all"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>
                );
              })}
              {cachedFiles.length === 0 && <p className="text-[10px] text-zinc-500 italic">No cached files.</p>}
            </div>
            <button
              onClick={handleClearCache}
              disabled={isProcessing}
              className={cn(
                "w-full py-1.5 border rounded-lg text-[10px] font-bold uppercase tracking-widest transition-colors",
                darkMode ? "border-zinc-800 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300" : "border-zinc-200 text-zinc-400 hover:bg-zinc-50 hover:text-zinc-600"
              )}
            >
              Purge All Cache
            </button>
          </div>
          
          {processStatus && !isProcessing && processStatus !== 'Error occurred.' && (
            <p className="text-xs text-center text-emerald-600 font-medium">{processStatus}</p>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col h-screen overflow-hidden">
        <div className={cn(
          "p-6 border-b transition-colors duration-300",
          darkMode ? "bg-zinc-900 border-zinc-800" : "bg-white border-zinc-200"
        )}>
          <div className="max-w-3xl mx-auto">
            <div className="relative">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                placeholder="Ask a question about your document..."
                className={cn(
                  "w-full pl-12 pr-24 py-4 border rounded-2xl text-base focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 shadow-sm transition-colors",
                  darkMode ? "bg-zinc-800 border-zinc-700 text-zinc-100" : "bg-zinc-50 border-zinc-200 text-zinc-900"
                )}
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
          <div className="max-w-full mx-auto" id="results-container">
            {Object.keys(results).length > 0 ? (
              <div className="space-y-12">
                <div className={cn(
                  "flex items-center justify-between p-4 rounded-xl border shadow-sm transition-colors",
                  darkMode ? "bg-zinc-900 border-zinc-800" : "bg-white border-zinc-200"
                )}>
                  <div>
                    <h3 className={cn("text-sm font-bold", darkMode ? "text-zinc-100" : "text-zinc-900")}>Query: {query}</h3>
                    <p className="text-xs text-zinc-500">File: {file?.name} | Generated at: {new Date().toLocaleString()}</p>
                  </div>
                  <button
                    onClick={handleExportPDF}
                    className="px-4 py-2 bg-emerald-600 text-white rounded-lg text-sm font-medium hover:bg-emerald-700 transition-colors flex items-center gap-2"
                  >
                    Download PDF Report
                  </button>
                </div>
                {selectedModels.map(model => results[model] && (
                  <div key={model} className="space-y-8">
                    <div className={cn(
                      "flex items-center justify-between border-b-2 pb-4 transition-colors",
                      darkMode ? "border-zinc-800" : "border-zinc-900"
                    )}>
                      <h2 className={cn("text-2xl font-bold", darkMode ? "text-zinc-100" : "text-zinc-900")}>{model}</h2>
                    </div>

                    {/* Answers Comparison Grid */}
                    <div className="space-y-8">
                      {selectedLLMModels.map(llm => (
                        <div key={llm} className="space-y-4">
                          <div className="flex items-center gap-2 border-l-4 border-indigo-500 pl-3">
                            <h3 className={cn("text-lg font-bold", darkMode ? "text-zinc-200" : "text-zinc-800")}>{llm}</h3>
                          </div>
                          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                            {selectedKValues.map(k => {
                              const result = results[model].answers[llm]?.[k];
                              return (
                                <div key={k} className="space-y-4">
                                    <div className="flex items-center justify-between">
                                      <div className="flex items-center gap-2">
                                        <span className="px-3 py-1 bg-indigo-600 text-white text-xs font-bold rounded-full">K = {k}</span>
                                        <h4 className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Answer</h4>
                                      </div>
                                      <div className="flex flex-col items-end gap-1">
                                        {result?.usage && (
                                          <span className="text-[10px] text-zinc-400">
                                            Tokens: {result.usage.totalTokenCount}
                                          </span>
                                        )}
                                        {result?.evaluation && (
                                          <div className="flex gap-2">
                                            <span className={cn(
                                              "text-[9px] font-bold px-1.5 py-0.5 rounded border",
                                              result.evaluation.faithfulness >= 8 ? "bg-emerald-50 border-emerald-200 text-emerald-700" : "bg-amber-50 border-amber-200 text-amber-700"
                                            )}>
                                              Faith: {result.evaluation.faithfulness}/10
                                            </span>
                                            <span className={cn(
                                              "text-[9px] font-bold px-1.5 py-0.5 rounded border",
                                              result.evaluation.relevance >= 8 ? "bg-indigo-50 border-indigo-200 text-indigo-700" : "bg-amber-50 border-amber-200 text-amber-700"
                                            )}>
                                              Rel: {result.evaluation.relevance}/10
                                            </span>
                                          </div>
                                        )}
                                      </div>
                                    </div>
                                    {result?.evaluation?.reasoning && (
                                      <p className="text-[10px] text-zinc-500 italic bg-zinc-50 p-2 rounded-lg border border-zinc-100">
                                        Eval Reasoning: {result.evaluation.reasoning}
                                      </p>
                                    )}
                                  <div className={cn(
                                    "p-6 rounded-2xl border-2 shadow-sm min-h-[200px] transition-colors overflow-y-auto max-h-[500px]",
                                    darkMode ? "bg-zinc-900 border-zinc-800" : "bg-white border-indigo-100"
                                  )}>
                                    <div className="space-y-6">
                                      {result?.history.map((msg: any, idx: number) => {
                                        // Skip the first user prompt as it contains the system prompt and context
                                        if (msg.role === 'user' && idx === 0) return null;
                                        
                                        return (
                                          <div key={idx} className={cn(
                                            "flex flex-col gap-1",
                                            msg.role === 'user' ? "items-end" : "items-start"
                                          )}>
                                            <span className="text-[9px] font-bold uppercase tracking-widest text-zinc-500">
                                              {msg.role === 'user' ? 'You' : 'Assistant'}
                                            </span>
                                            <div className={cn(
                                              "px-4 py-2 rounded-2xl text-sm max-w-[90%]",
                                              msg.role === 'user' 
                                                ? (darkMode ? "bg-indigo-600 text-white" : "bg-indigo-500 text-white")
                                                : (darkMode ? "bg-zinc-800 text-zinc-200" : "bg-zinc-100 text-zinc-800")
                                            )}>
                                              <div className={cn(
                                                "leading-relaxed prose prose-sm max-w-none",
                                                msg.role === 'user' ? "prose-invert" : (darkMode ? "prose-invert" : "prose-zinc")
                                              )}>
                                                <Markdown>
                                                  {msg.parts[0].text}
                                                </Markdown>
                                              </div>
                                            </div>
                                          </div>
                                        );
                                      })}
                                      {/* If it's the first turn, just show the text */}
                                      {(!result?.history || result.history.length === 0) && (
                                        <div className={cn(
                                          "leading-relaxed prose prose-sm max-w-none",
                                          darkMode ? "text-zinc-300 prose-invert" : "text-zinc-700 prose-zinc"
                                        )}>
                                          <Markdown>{result?.text || 'Processing...'}</Markdown>
                                        </div>
                                      )}
                                      {/* If history exists, the last model message is already in the map above */}
                                    </div>
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      ))}
                    </div>

                    {/* Retrieved Chunks */}
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <h3 className="text-sm font-medium text-zinc-500 uppercase tracking-wider">Retrieved Context (Top 10)</h3>
                          {useReRanking && (
                            <span className="text-[9px] font-bold text-emerald-600 bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-100">RE-RANKED</span>
                          )}
                        </div>
                        <p className="text-[10px] text-zinc-400">Threshold: {threshold.toFixed(2)} | Chunks meeting threshold: {results[model].retrievedChunks.filter(c => c.score >= threshold).length}</p>
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-3">
                        {(results[model].reRankedChunks || results[model].retrievedChunks).map((chunk, i) => {
                          const isAboveThreshold = chunk.score >= threshold;
                          const isInAnyK = isAboveThreshold && selectedKValues.some(k => i < k);
                          const kStatus = isAboveThreshold ? selectedKValues.filter(k => i < k) : [];
                          
                          // Find original rank if re-ranked
                          const originalRank = results[model].reRankedChunks 
                            ? results[model].retrievedChunks.findIndex(c => c.text === chunk.text) + 1
                            : i + 1;

                          return (
                            <div 
                              key={i} 
                              className={cn(
                                "p-3 rounded-xl border shadow-sm transition-all text-xs",
                                isInAnyK ? (darkMode ? "bg-indigo-500/10 border-indigo-500/30 ring-1 ring-indigo-500/20" : "bg-white border-indigo-200 ring-1 ring-indigo-500/5") : (darkMode ? "bg-zinc-900 border-zinc-800 opacity-40" : "bg-white border-zinc-200 opacity-40"),
                                !isAboveThreshold && "grayscale"
                              )}
                            >
                              <div className="flex flex-col gap-1.5 mb-2">
                                <div className="flex items-center justify-between">
                                  <div className="flex items-center gap-1">
                                    <span className="font-bold text-zinc-400">#{i + 1}</span>
                                    {useReRanking && originalRank !== (i + 1) && (
                                      <span className="text-[9px] text-zinc-400 line-through">({originalRank})</span>
                                    )}
                                  </div>
                                  <span className={cn(
                                    "font-mono px-1.5 py-0.5 rounded",
                                    isAboveThreshold ? "text-indigo-600 bg-indigo-50" : "text-zinc-400 bg-zinc-100"
                                  )}>
                                    {chunk.score.toFixed(3)}
                                  </span>
                                </div>
                                <div className="flex flex-wrap gap-1">
                                  {kStatus.map(k => (
                                    <span key={k} className="text-[9px] font-bold text-white bg-indigo-500 px-1 rounded">K{k}</span>
                                  ))}
                                  {!isAboveThreshold && (
                                    <span className="text-[9px] font-bold text-zinc-400 bg-zinc-100 px-1 rounded">BELOW THRESHOLD</span>
                                  )}
                                </div>
                              </div>
                              <p className={cn(
                                "line-clamp-6 leading-tight",
                                darkMode ? "text-zinc-400" : "text-zinc-600"
                              )}>{chunk.text}</p>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-zinc-400 space-y-4 py-20">
                <div className={cn(
                  "w-16 h-16 rounded-2xl flex items-center justify-center",
                  darkMode ? "bg-zinc-900" : "bg-zinc-100"
                )}>
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
