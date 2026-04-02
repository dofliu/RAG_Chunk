# 📝 待辦事項 (TODO)

以下是目前專案中可以立即著手進行的改進項目：

## 🔴 高優先級 (High Priority)
- [x] **報告輸出功能**：已實作將比較結果整理成 PDF 輸出下載。
- [x] **多模型並列回答**：支援同時選擇多個 LLM 模型 (Gemini 3.1 Flash vs Pro) 並列比較。
- [x] **自定義 System Prompt**：已加入 Textarea，讓使用者自定義回答指令。

## 🟡 中優先級 (Medium Priority)
- [x] **顯示 Token 數**：在回答下方顯示該次 RAG 消耗的 Token 數量。
- [x] **相似度閾值過濾**：加入 Slider，過濾掉相似度低於特定分數的段落。
- [x] **多輪對話支援**：在當前對話基礎上繼續追問，並將前文納入 Context。

## 🟢 低優先級 (Low Priority)
- [x] **UI 亮/暗模式切換**：提供更好的視覺體驗。
- [x] **支援更多檔案格式**：如 Markdown (.md), JSON (.json)。
- [x] **本地向量庫管理**：列出所有已儲存在 IndexedDB 中的檔案，並提供刪除單一檔案快取的功能。

---

## 💡 下一步討論建議
1. **RAG 評估**：已實作自動化評估指標（Faithfulness, Answer Relevance）。
2. **重排序 (Re-ranking)**：已實作 LLM Re-rank 功能，優化檢索排序。
3. **切片策略優化**：已加入「語義切片 (Semantic Chunking)」與「段落切片」選項。
