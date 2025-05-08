import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel

#############################################
# 1️⃣ Page Configuration & Custom Styling
#############################################
st.set_page_config(page_title="Toxic Comment Classifier", page_icon="🤖", layout="centered")
st.markdown("""
    <style>
        body {
            background-color: #f4f4f4;
            font-family: 'Arial', sans-serif;
        }
        .main-title {
            text-align: center;
            font-size: 36px;
            color: #4A90E2;
            font-weight: bold;
        }
        .sub-title {
            text-align: center;
            font-size: 18px;
            color: #333333;
            margin-bottom: 30px;
        }
        .stTextArea textarea {
            background-color: #ffffff;
            border-radius: 10px;
            border: 2px solid #4A90E2;
            padding: 10px;
            font-size: 16px;
        }
        .stButton button {
            background-color: #4A90E2;
            color: white;
            border-radius: 10px;
            font-size: 18px;
            padding: 8px 20px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

#############################################
# 2️⃣ Define Labels and Model Architecture
#############################################
label_cols = ['Toxic', 'Severe Toxic', 'Obscene', 'Threat', 'Insult', 'Identity Hate']

class ToxicityClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.emoji_layer = nn.Linear(3, 64)
        self.classifier = nn.Linear(768 + 64, 6)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, attention_mask, emoji_feats):
        # Process text through DistilBERT
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = bert_out.last_hidden_state[:, 0, :]  # Use [CLS] token output
        
        # Process emoji features and boost their influence
        emoji_out = self.emoji_layer(emoji_feats)
        emoji_out = emoji_out * 5.0  # Increase emoji influence
        
        # Concatenate text and emoji features
        combined = self.dropout(torch.cat([pooled, emoji_out], dim=1))
        return self.classifier(combined)

#############################################
# 3️⃣ Load the Trained Model and Tokenizer
#############################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ToxicityClassifier().to(device)
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.eval()

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

#############################################
# 4️⃣ Define Emoji Toxicity Mapping
#############################################
# Updated toxicity values: higher magnitude increases effect.
emoji_toxicity = {
    "😊": -3.0, "😡": 3.0, "🤢": 2.5, "😱": 1.5, "😈": 3.0,
    "❤️": -4.0, "😢": 1.5, "😂": -3.0, "👍": -1.0, "💀": 2.8
}

#############################################
# 5️⃣ Emoji Feature Extraction Function
#############################################
def extract_emoji_features(text):
    """
    Extracts three features from the text:
      1. Count of emojis in the text.
      2. Average toxicity score of the emojis.
      3. Maximum toxicity score among the emojis.
    """
    emojis = [c for c in text if c in emoji_toxicity]
    if not emojis:
        return np.array([0, 0, 0], dtype='float32')
    toxicities = [emoji_toxicity[e] for e in emojis]
    return np.array([
        len(emojis),
        np.mean(toxicities),
        max(toxicities)
    ], dtype='float32')

#############################################
# 6️⃣ Streamlit App Layout and User Input
#############################################
st.markdown('<h1 class="main-title">🚀 Toxic Comment & Emoji Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Enter your comment, select emojis, and classify its toxicity.</p>', unsafe_allow_html=True)

comment_text = st.text_area("📝 Enter Comment", placeholder="Type your comment here...")
selected_emojis = st.multiselect("😀 Select Emojis", options=list(emoji_toxicity.keys()))

#############################################
# 7️⃣ Inference and Display of Results
#############################################
if st.button("🔍 Classify Now"):
    # Combine text with selected emojis
    full_text = comment_text + " " + "".join(selected_emojis)
    
    # Tokenize the combined text
    encoding = tokenizer(
        full_text, truncation=True, padding='max_length', max_length=128, return_tensors='pt'
    )
    
    # Extract emoji features and convert to tensor
    emoji_feats = extract_emoji_features(full_text)
    emoji_feats = torch.tensor(emoji_feats, dtype=torch.float32).unsqueeze(0)
    
    # Move tensors to the appropriate device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    emoji_feats = emoji_feats.to(device)
    
    # Run inference with the model
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, emoji_feats=emoji_feats)
        probabilities = torch.sigmoid(outputs).squeeze().cpu().numpy()
    
    st.subheader("📊 Prediction Results:")
    
    # Display each prediction with the value and a custom side-by-side line graph
    for label, prob in zip(label_cols, probabilities):
        col_label, col_bar = st.columns([2, 1])
        with col_label:
            st.write(f"**{label}:** {prob * 100:.2f}%")
        with col_bar:
            st.markdown(f"""
            <div style="background-color: #e0e0e0; width: 100%; height: 20px; border-radius: 5px;">
                <div style="background-color: #4A90E2; width: {prob * 100}%; height: 20px; border-radius: 5px;"></div>
            </div>
            """, unsafe_allow_html=True)
