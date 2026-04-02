# Embedding & RAG Compare Lab

這是一個基於 Google Gemini API 的實驗性專案，旨在幫助開發者與研究人員比較不同 **Embedding 模型** 以及 **RAG (Retrieval-Augmented Generation) 策略** 對回答品質的影響。

## 🚀 主要功能

- **多格式文件支援**：支援上傳 PDF, DOCX, Excel, TXT 等文件。
- **自定義切片策略**：可調整 Chunk Size 與 Overlap 參數。
- **模型並列比較**：同時使用多個 Gemini Embedding 模型進行向量化。
- **多 K 值實驗 (Side-by-Side)**：
    - 支援同時選擇多個 K 值（如 3, 5, 7, 10）。
    - 並列顯示不同 K 值下的 LLM 回答結果，觀察上下文長度對回答精準度的影響。
- **高效本地快取**：使用 IndexedDB (idb-keyval) 儲存向量資料，避免重複處理相同檔案，節省 API 額度。
- **視覺化檢索結果**：
    - 列出 Top 10 相關段落。
    - 標示哪些段落被納入當前 K 值的上下文。
    - 顯示相似度分數 (Cosine Similarity)。
- **引用標註**：LLM 在回答時會自動標註引用來源（如 `[Chunk 1]`）。

## 🛠️ 技術棧

- **Frontend**: React 18, Vite, Tailwind CSS
- **AI SDK**: `@google/genai` (Gemini 3.1 Flash/Pro)
- **Storage**: `idb-keyval` (IndexedDB)
- **Parsing**: `pdfjs-dist`, `mammoth`, `xlsx`
- **UI Components**: Lucide React, Tailwind Typography

## 📖 使用說明

1. **上傳檔案**：在左側面板選擇您的文件。
2. **設定參數**：調整切片大小、重疊度，並選擇要比較的 Embedding 模型。
3. **處理文件**：點擊 "Process Document" 進行向量化。
4. **選擇 K 值**：在 "Compare K Values" 中選取您想比較的數值（可多選）。
5. **提問**：在上方搜尋框輸入問題，即可看到不同模型與 K 值下的回答差異。

---

## 🔑 環境變數

請確保在 `.env` 中設定以下變數：
```env
GEMINI_API_KEY=您的_Gemini_API_Key
```
