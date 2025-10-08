import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights
from config import num_classes

class CrossModalAttention(nn.Module):
    def __init__(self, img_dim, txt_dim, attn_dim=128):
        super().__init__()
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, attn_dim),
            nn.LayerNorm(attn_dim)
        )
        self.txt_proj = nn.Sequential(
            nn.Linear(txt_dim, attn_dim),
            nn.LayerNorm(attn_dim)
        )
        self.attn = nn.MultiheadAttention(attn_dim, num_heads=4, batch_first=True)

    def forward(self, img_feats, txt_feats):
        Q = self.txt_proj(txt_feats)
        K = V = self.img_proj(img_feats)
        attn_output, _ = self.attn(Q, K, V)
        return attn_output

class DepthwiseSeparableConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.depthwise = nn.Conv2d(
            input_dim + hidden_dim, 
            input_dim + hidden_dim, 
            kernel_size=3, 
            padding=1, 
            groups=input_dim + hidden_dim
        )
        self.pointwise = nn.Conv2d(
            input_dim + hidden_dim, 
            4 * hidden_dim, 
            kernel_size=1
        )

    def forward(self, x, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([x, h_cur], dim=1)
        gates = self.pointwise(self.depthwise(combined))
        i, f, o, g = torch.split(gates, gates.size(1) // 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class StableCNNLSTM(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(StableCNNLSTM, self).__init__()

        self.img_cnn = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        for param in self.img_cnn.features[:15].parameters():
            param.requires_grad = False
        classifier_children = list(self.img_cnn.classifier.children())
        self.img_cnn.classifier = nn.Sequential(*classifier_children[:-1])
        self.img_proj = nn.Sequential(
            nn.Linear(1280, 256),
            nn.LayerNorm(256),
            nn.Dropout(0.3)
        )

        self.eye_txt_bilstm = nn.LSTM(
            input_size=5,
            hidden_size=64,
            bidirectional=True,
            batch_first=True
        )
        self.eye_txt_proj = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.Dropout(0.2)
        )

        self.flight_txt_bilstm = nn.LSTM(
            input_size=3, 
            hidden_size=64, 
            bidirectional=True, 
            batch_first=True
        )
        self.flight_txt_proj = nn.Sequential(
            nn.Linear(128, 64),  
            nn.LayerNorm(64),
            nn.Dropout(0.2)
        )

        self.cross_attn = CrossModalAttention(img_dim=256, txt_dim=64 + 64)
        self.attn_norm = nn.LayerNorm(128)

        self.convlstm = DepthwiseSeparableConvLSTMCell(input_dim=128, hidden_dim=128)
        self.lstm_norm = nn.LayerNorm(128)
        self.lstm_dropout = nn.Dropout(0.4)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_normal_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
                        param.data[m.hidden_size:2*m.hidden_size] = 1.0

    def forward(self, img_seq, eye_txt_seq, flight_txt_seq):
        batch_size, seq_len = img_seq.size(0), img_seq.size(1)

        img_feats = self.img_cnn(img_seq.view(-1, 3, 224, 224))
        img_feats = self.img_proj(img_feats).view(batch_size, seq_len, -1)

        eye_txt_feats, _ = self.eye_txt_bilstm(eye_txt_seq)
        eye_txt_feats = self.eye_txt_proj(eye_txt_feats)

        flight_txt_feats, _ = self.flight_txt_bilstm(flight_txt_seq)
        flight_txt_feats = self.flight_txt_proj(flight_txt_feats)

        combined_txt_feats = torch.cat([eye_txt_feats, flight_txt_feats], dim=2)

        fused_feats = self.cross_attn(img_feats, combined_txt_feats)
        fused_feats = self.attn_norm(fused_feats)
        fused_feats = self.lstm_dropout(fused_feats)

        spatial_feats = fused_feats.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)

        h = torch.zeros(batch_size, 128, 1, 1, device=img_seq.device)
        c = torch.zeros(batch_size, 128, 1, 1, device=img_seq.device)

        for t in range(seq_len):
            h, c = self.convlstm(spatial_feats[:, :, t, :, :], (h, c))
            h = h.permute(0, 2, 3, 1)
            h = self.lstm_norm(h).permute(0, 3, 1, 2)
            h = self.lstm_dropout(h)

        h = h.squeeze(-1).squeeze(-1)
        return self.classifier(h)