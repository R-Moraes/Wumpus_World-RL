import numpy as np

MAPS = {
    "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
}

map_name="4x4"
desc = MAPS[map_name]
print(desc)
desc2 = desc = np.asarray(desc, dtype="c")
desc = desc2.tolist()
desc = [[c.decode("utf-8") for c in line] for line in desc]
