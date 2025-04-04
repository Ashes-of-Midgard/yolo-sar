import torch
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.nn.model import Model

old_weights_path = 'yolo12m.pt'
new_model_cfg = 'yolo12m_new.yaml'
save_path = 'yolo12m_new_init.pt'

print(f"Loading original model from: {old_weights_path}")
old_model, _ = attempt_load_one_weight(old_weights_path)

print(f"Creating new model from config: {new_model_cfg}")
new_model = Model(new_model_cfg, ch=3, verbose=False)

def transfer_weights(old_model, new_model):
    old_state = old_model.state_dict()
    new_state = new_model.state_dict()
    transferred = 0
    for k in new_state.keys():
        if k in old_state and old_state[k].shape == new_state[k].shape:
            new_state[k] = old_state[k]
            transferred += 1
        else:
            print(f"Skipping: {k} -- {new_state[k].shape} (old: {old_state.get(k, 'N/A')})")
    new_model.load_state_dict(new_state)
    print(f"\n✅ Transferred {transferred}/{len(new_state)} parameters.")

transfer_weights(old_model, new_model)

torch.save(new_model.state_dict(), save_path)
print(f"\n✅ Saved new initialized model weights to: {save_path}")
