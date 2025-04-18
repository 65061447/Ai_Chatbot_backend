import pandas as pd
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter

# 1. Load CSV
df = pd.read_csv("allattractions_with_context.csv")
print("✅ CSV loaded with shape:", df.shape)

# 2. ตรวจสอบว่ามี column 'context' หรือไม่
if "context" not in df.columns:
    raise ValueError("❌ The CSV must have a 'context' column")

# 3. รวมข้อความจากคอลัมน์ 'context'
context_list = df["context"].dropna().astype(str).tolist()
print(f"✅ Loaded {len(context_list)} non-empty context rows")

all_text = "\n".join(context_list)
print("📝 Sample context text (first 500 chars):\n", all_text[:500])

# 4. แบ่งข้อความออกเป็น chunks
splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=100
)
chunks = splitter.split_text(all_text)
print(f"🧩 Text split into {len(chunks)} chunks")
print("📎 Example chunk:\n", chunks[0][:300])

# 5. สร้างเวกเตอร์จาก chunks ด้วยโมเดล BAAI/bge-m3
print("🔄 Encoding chunks with model 'BAAI/bge-m3'...")
model = SentenceTransformer("BAAI/bge-m3")
embeddings = model.encode(chunks, show_progress_bar=True)
print("✅ Embedding shape:", np.array(embeddings).shape)

# 6. สร้าง FAISS index และเพิ่ม embeddings เข้าไป
index = faiss.IndexFlatL2(embeddings[0].shape[0])
index.add(np.array(embeddings))
print("📦 FAISS index created and populated")

# 7. บันทึก index และ chunks
faiss.write_index(index, "vector_index.faiss")
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print(f"\n✅ Vector store created successfully with {len(chunks)} chunks")
