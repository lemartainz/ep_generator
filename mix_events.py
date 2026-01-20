#%%
def read_lund_blocks(filename):
    blocks = []
    with open(filename) as f:
        current = []
        for line in f:
            line = line.rstrip()
            if line == "":
                if current:
                    blocks.append("\n".join(current))
                    current = []
                continue
            current.append(line)
        if current:
            blocks.append("\n".join(current))
    return blocks

signal = read_lund_blocks("events_signal.lund")
bkg1   = read_lund_blocks("events_back_pi0.lund")
bkg2   = read_lund_blocks("events_back_2g.lund")
bkg3 = read_lund_blocks("events_back_g.lund")
bkg4 = read_lund_blocks("events_back_p_pipi.lund")

# %%
all_events = signal + bkg1 + bkg2 + bkg3 + bkg4

import random

random.shuffle(all_events)

with open("events_mixed.lund", "w") as f:
    for block in all_events:
        f.write(block)
        f.write("\n\n")
# %%
