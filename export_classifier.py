import torch
import torch.nn as nn
import numpy as np

# 1. Define Model Classes (Same as before)
class LSTMNet(nn.Module):
    def __init__(self):
        super(LSTMNet, self).__init__()
        self.lstm1 = nn.LSTM(1024, 256, bidirectional=True)
        self.lstm2 = nn.LSTM(2 * 256, 32, bidirectional=True)
        self.fc_ha = nn.Linear(2 * 32, 100)
        self.fc_1 = nn.Linear(100, 1)
        self.sftmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1, _ = self.lstm1(x)
        x2, _ = self.lstm2(x1)
        ht = x2[-1]
        ht = torch.unsqueeze(ht, 0)
        ha = torch.tanh(self.fc_ha(ht))
        alp = self.fc_1(ha)
        al = self.sftmax(alp)
        T = ht.size(1)
        batch_size = ht.size(0)
        D = ht.size(2)
        c = torch.bmm(al.view(batch_size, 1, T), ht.view(batch_size, T, D))
        c = torch.squeeze(c, 0)
        return c

class MSA_DAT_Net(nn.Module):
    def __init__(self, model1, model2):
        super(MSA_DAT_Net, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.att1 = nn.Linear(2 * 32, 100)
        self.att2 = nn.Linear(100, 1)
        self.bsftmax = nn.Softmax(dim=1)
        self.lang_classifier = nn.Linear(2 * 32, 12, bias=True)

    def forward(self, x1, x2):
        u1 = self.model1(x1)
        u2 = self.model2(x2)
        ht_u = torch.cat((u1, u2), dim=0)
        ht_u = torch.unsqueeze(ht_u, 0)
        ha_u = torch.tanh(self.att1(ht_u))
        alp = torch.tanh(self.att2(ha_u))
        al = self.bsftmax(alp)
        Tb = ht_u.size(1)
        batch_size = ht_u.size(0)
        D = ht_u.size(2)
        u_vec = torch.bmm(al.view(batch_size, 1, Tb), ht_u.view(batch_size, Tb, D))
        u_vec = torch.squeeze(u_vec, 0)
        lang_output = self.lang_classifier(u_vec)
        return lang_output

# 2. Load Model
print("Loading Classifier Model...")
model1 = LSTMNet()
model2 = LSTMNet()
model = MSA_DAT_Net(model1, model2)
model_path = './model/ZWSSL_train_SpringData_13June2024_e3.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
model.eval()

# 3. Dummy Inputs
dummy_x1 = torch.randn(20, 1, 1024) 
dummy_x2 = torch.randn(17, 1, 1024)

# 4. Export
print("Exporting Classifier to ONNX...")
output_file = "lid_classifier.onnx"

torch.onnx.export(
    model,
    (dummy_x1, dummy_x2),
    output_file,
    export_params=True,
    opset_version=14,
    input_names=['x1', 'x2'],
    output_names=['lang_output'],
    dynamic_axes={
        # CRITICAL FIX: Use UNIQUE names for the dynamic axes
        'x1': {1: 'num_windows_x1'}, 
        'x2': {1: 'num_windows_x2'},
        'lang_output': {0: 'batch_size'} 
    }
)

print(f"Success! Model saved to {output_file}")