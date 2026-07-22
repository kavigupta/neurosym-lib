import json
for split in ('train', 'val'):
    filename = f"CLEVR_{split}_2_remove.json"
    new_filename = f"CLEVR_{split}_2_remove_easy.json"
    with open(filename, "r") as f:
        data = json.load(f)
        if split == 'train':
            data['questions'] = data['questions'][2:7]
        else:
            data['questions'] = data['questions'][2:7]
    with open(new_filename, 'w') as f:
        json.dump(data,f )