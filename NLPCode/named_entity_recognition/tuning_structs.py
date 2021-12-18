from enum import IntEnum, Flag


class Privacy(Flag):
    No_Privacy_Preserving = False
    Privacy_Preserving = True


class TrainingBERT(IntEnum):
    OnlyEmbeds = 1
    Freeze_Total = 0
    Train_All = 2
    Train_Last_1 = -1
    Train_Last_2 = -2
    Train_Last_3 = -3
    Train_Last_4 = -4
    Train_Last_5 = -5
    Train_Last_6 = -6
    Train_Last_7 = -7
    Train_Last_8 = -8
    Train_Last_9 = -9
    Train_Last_10 = -10
    Train_Last_11 = -11


class Tuning:
    def __init__(self, privacy_preserving: Privacy, training_bert: TrainingBERT, model_type):
        self.privacy = privacy_preserving
        self.training_bert = training_bert

        self.num_layers = 12 if model_type == 'bert-base-cased' else 24

        self.embeddings = ["embeddings.word_embeddings.weight",
                           "embeddings.position_embeddings.weight",
                           "embeddings.token_type_embeddings.weight",
                           "embeddings.LayerNorm.weight",
                           "embeddings.LayerNorm.bias"]

        self.pooler = ["pooler.dense.weight", "pooler.dense.bias"]

        self.encoder = [[f"encoder.layer.{i}.attention.self.query.weight",
                         f"encoder.layer.{i}.attention.self.query.bias",
                         f"encoder.layer.{i}.attention.self.key.weight",
                         f"encoder.layer.{i}.attention.self.key.bias",
                         f"encoder.layer.{i}.attention.self.value.weight",
                         f"encoder.layer.{i}.attention.self.value.bias",
                         f"encoder.layer.{i}.attention.output.dense.weight",
                         f"encoder.layer.{i}.attention.output.dense.bias",
                         f"encoder.layer.{i}.attention.output.LayerNorm.weight",
                         f"encoder.layer.{i}.attention.output.LayerNorm.bias",
                         f"encoder.layer.{i}.intermediate.dense.weight",
                         f"encoder.layer.{i}.intermediate.dense.bias",
                         f"encoder.layer.{i}.output.dense.weight",
                         f"encoder.layer.{i}.output.dense.bias",
                         f"encoder.layer.{i}.output.LayerNorm.weight",
                         f"encoder.layer.{i}.output.LayerNorm.bias"] for i in range(self.num_layers)]

        if self.training_bert < 0:
            self.freeze_array = self.embeddings.copy()
            self.freeze_array.extend(self.pooler)

            for arr in self.encoder[:training_bert]:
                self.freeze_array.extend(arr)
        # because we only want to freez the bert model, not the linear layer at the end
        elif self.training_bert == TrainingBERT.Freeze_Total:
            self.freeze_array = self.embeddings.copy()
            self.freeze_array.extend(self.pooler)
            for arr in self.encoder:
                self.freeze_array.extend(arr)

