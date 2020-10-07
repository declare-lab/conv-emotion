import torch


def prepare_position_embeddings(opt, encoder_vocab, sequences):
    vocab_size = len(encoder_vocab)
    num_positions = sequences.size(-2)
    position_embeddings = torch.LongTensor(
        range(vocab_size, vocab_size + num_positions)).to(sequences.device)
    sequences = sequences.repeat(1, 1, 2)
    sequences[:, :, 1] = position_embeddings
    return sequences

