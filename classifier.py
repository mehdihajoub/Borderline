import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderLayer(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden_size, dropout_rate):
        super(EncoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, dropout=dropout_rate)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, ff_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_hidden_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Multi-head Attention
        attn_output, _ = self.attention(src, src, src, attn_mask=src_mask,
                                        key_padding_mask=src_key_padding_mask)
        src = src + self.dropout(attn_output)
        src = self.norm1(src)

        # Feed Forward
        ff_output = self.feed_forward(src)
        src = src + self.dropout(ff_output)
        src = self.norm2(src)

        return src

class BERTModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, ff_hidden_size, num_layers, dropout_rate, max_position_embeddings, num_segments):
        super(BERTModel, self).__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.segment_embedding = nn.Embedding(num_segments, embed_size)
        self.position_embedding = nn.Embedding(max_position_embeddings, embed_size)

        self.layers = nn.ModuleList([EncoderLayer(embed_size, num_heads, ff_hidden_size, dropout_rate) for _ in range(num_layers)])
        self.pooler = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout_rate)

        # Classification head for binary classification
        self.classifier = nn.Linear(embed_size, 1)

    def forward(self, x, segments):
        seq_length = x.size(1)
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=x.device).unsqueeze(0)

        token_embeddings = self.token_embedding(x)
        segment_embeddings = self.segment_embedding(segments)
        position_embeddings = self.position_embedding(position_ids)

        embeddings = token_embeddings + segment_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)

        # Pass embeddings through each of the encoder layers in sequence
        for layer in self.layers:
            embeddings = layer(embeddings)

        # Pooling operation on the encoder's first token's output
        pooled_output = self.pooler(embeddings[:, 0])
        pooled_output = torch.tanh(pooled_output)

        # Pass the pooled output through the classifier to get the logit
        logit = self.classifier(pooled_output)

        # Apply sigmoid to convert the logit to a probability for binary classification
        probability = torch.sigmoid(logit)

        return probability


# Example of initializing BERT model
vocab_size = 30522             # Size of vocabulary for BERT base
embed_size = 768               # Embedding size for BERT base
num_heads = 12                 # Number of attention heads
ff_hidden_size = 3072          # Size of the feed-forward layers
num_layers = 12                # Number of encoder layers
dropout_rate = 0.1             # Dropout rate
max_position_embeddings = 512  # Maximum sequence length
num_segments = 2               # Number of sentence segments


bert_model = BERTModel(vocab_size, embed_size, num_heads,
                        ff_hidden_size, num_layers, dropout_rate,
                        max_position_embeddings, num_segments)

inp = 'Bonjour Je Suis Content'

output = bert_model (inp)

print(output)