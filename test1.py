import sys, os, torch
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([os.path.join(ROOT_DIR, "src"), os.path.join(ROOT_DIR, "src", "models"), os.path.join(ROOT_DIR, "src", "stats")])
from fusion_model import MultimodalSportsModel
from main import _generate_plots

device = torch.device("cpu")
model = MultimodalSportsModel().to(device)
model.load_state_dict(torch.load("src/models/multimodal_v1.pth", map_location=device, weights_only=True))

_generate_plots(model, "data/processed/test_stats.csv", "data/news_embeddings.pt", "data/historical_embeddings.pt")