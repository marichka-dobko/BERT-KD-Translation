from transformers import Seq2SeqTrainer
from torch.nn import CrossEntropyLoss
from KD_training.loss import BertKDLossCompute

class GeneratorWrapper(object):
    def __init__(self, proj):
        self.proj = proj

    def __call__(self, input_):
        output = self.proj(input_).float()
        return output


class BertKDTrainer(Seq2SeqTrainer):
    def __init__(self, vocab, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tgt_field = dict(vocab)["tgt"].base_field
        padding_idx = tgt_field.vocab.stoi[tgt_field.pad_token]
        generator = GeneratorWrapper(self.model.generator[0])
        self.train_loss = BertKDLossCompute(generator,
                                           padding_idx,
                                           0.1,   # label_smoothing
                                           8)   # top-k

    def compute_loss(self, model, inputs, return_outputs=False):
        # labels = inputs.pop("labels")
        outputs = model(**inputs)
        # logits = outputs[0]
        # loss_fct = CrossEntropyLoss()
        # loss = loss_fct(logits, labels)

        tgt_out = inputs.tgt[1:, :, 0]
        loss, batch_stats = self.train_loss(
            outputs, inputs.bert_topk, tgt_out,
            'tokens',
            self.temperature, self.alpha)

        return (loss, outputs) if return_outputs else loss
