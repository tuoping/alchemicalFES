import re


def upgrade_state_dict(state_dict, prefixes=["encoder.sentence_encoder.", "encoder."], substiteby=""):
    """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
    pattern = re.compile("^" + "|".join(prefixes))
    state_dict = {pattern.sub(substiteby, name): param for name, param in state_dict.items()}
    return state_dict

