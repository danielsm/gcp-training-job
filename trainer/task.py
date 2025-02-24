import os
import argparse
import glob
import random
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.models import model_from_json
from tensorflow.keras.models import load_model
import math
from tensorflow.keras import Model
from tensorflow.keras.layers import Add, Dense, Dropout, Embedding, GlobalAveragePooling1D, Input, Lambda, Layer, LayerNormalization, MultiHeadAttention, Flatten, Conv2D, Reshape
import pickle  
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam


def _parse_function(proto):
    """Parse a single TFRecord example"""
    feature_description = {
        'q10': tf.io.FixedLenFeature([], tf.string),
        'q50': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(proto, feature_description)

    q10 = tf.io.decode_raw(example['q10'], tf.float32)
    q50 = tf.io.decode_raw(example['q50'], tf.float32)

    q10 = tf.reshape(q10, [224, 224, 3])
    q50 = tf.reshape(q50, [224, 224, 3])

    return q10, q50

def load_compressed_tfrecord_dataset(tfrecord_dir, batch_size=16):
    """Load and process compressed TFRecord shards efficiently"""
    tfrecord_files = tf.io.gfile.glob(f"{tfrecord_dir}/*.tfrecord")
    print(tfrecord_files)
    dataset = tf.data.TFRecordDataset(
        tfrecord_files,
        compression_type="GZIP",
        num_parallel_reads=tf.data.AUTOTUNE
    )

    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return dataset


class PatchExtractor(Layer):
    def __init__(self, patch_size, **kwargs):
        super(PatchExtractor, self).__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config
        
class PatchEncoder(Layer):
    def __init__(self, num_patches=196, projection_dim=768, **kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        w_init = tf.random_normal_initializer()
        class_token = w_init(shape=(1, self.projection_dim), dtype="float32")
        self.class_token = tf.Variable(initial_value=class_token, trainable=True)
        self.projection = Dense(units=self.projection_dim)
        self.position_embedding = Embedding(input_dim=self.num_patches+1, output_dim=self.projection_dim)

    def call(self, patch):
        batch = tf.shape(patch)[0]
        # reshape the class token embedins
        class_token = tf.tile(self.class_token, multiples = [batch, 1])
        class_token = tf.reshape(class_token, (batch, 1, self.projection_dim))
        # calculate patches embeddings
        patches_embed = self.projection(patch)
        patches_embed = tf.concat([patches_embed, class_token], 1)
        # calcualte positional embeddings
        positions = tf.range(start=0, limit=self.num_patches+1, delta=1)
        positions_embed = self.position_embedding(positions)
        # add both embeddings
        encoded = patches_embed + positions_embed
        return encoded
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim,
        })
        return config


class MLP(Layer):
    def __init__(self, hidden_features, out_features, dropout_rate=0.2, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.dropout_rate = dropout_rate
        self.dense1 = Dense(self.hidden_features, activation=tf.nn.gelu)
        self.dense2 = Dense(self.out_features)
        self.dropout = Dropout(self.dropout_rate)

    def call(self, x):
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        y = self.dropout(x)
        return y
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_features": self.hidden_features,
            "out_features": self.out_features,
            "dropout_rate": self.dropout_rate,
        })
        return config

    
class Block(Layer):
    def __init__(self, projection_dim, num_heads=4, dropout_rate=0.1):
        super(Block, self).__init__()
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.attn = MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=dropout_rate)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.mlp = MLP(projection_dim * 2, projection_dim, dropout_rate)
        self.attention_scores = None

    def call(self, x, return_attention=False):
        # Layer normalization 1.
        x1 = self.norm1(x) # encoded_patches
        # Create a multi-head attention layer.
        if return_attention:
            attention_output, attention_scores = self.attn(x1, x1, return_attention_scores=True)
            self.attention_scores = attention_scores
                # Skip connection 1.
            x2 = Add()([attention_output, x]) #encoded_patches
            # Layer normalization 2.
            x3 = self.norm2(x2)
            # MLP.
            x3 = self.mlp(x3)
            # Skip connection 2.
            y = Add()([x3, x2])
            return y, attention_scores
        else:
            attention_output = self.attn(x1, x1)
            # Skip connection 1.
            x2 = Add()([attention_output, x]) #encoded_patches
            # Layer normalization 2.
            x3 = self.norm2(x2)
            # MLP.
            x3 = self.mlp(x3)
            # Skip connection 2.
            y = Add()([x3, x2])
            return y

class TransformerEncoder(Layer):
    def __init__(self, projection_dim, num_heads=4, num_blocks=8, dropout_rate=0.5, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.num_blocks = num_blocks
        self.blocks = [Block(self.projection_dim, self.num_heads, self.dropout_rate) for _ in range(self.num_blocks)]
        self.norm = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(self.dropout_rate)
        self.attention_scores = []

    def call(self, x):
        # Create a [batch_size, projection_dim] tensor.
        #self.attention_scores = []
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        y = self.dropout(x)
        return y
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "projection_dim": self.projection_dim,
            "num_heads": self.num_heads,
            "num_blocks": self.num_blocks,
            "dropout_rate": self.dropout_rate,
        })
        return config


def create_vit_model(input_shape, num_patches, patch_size=8, projection_dim=192, num_blocks=12, num_heads=6, num_classes=0, dropout_rate=0.5):
    input = Input(shape=input_shape)
  
    patchExtractor = PatchExtractor(patch_size)
    patches = patchExtractor(input)

    patchEncoder = PatchEncoder(num_patches, projection_dim)
    patches_embed = patchEncoder(patches)

    transformers = TransformerEncoder(projection_dim, num_heads=num_heads, num_blocks=num_blocks)
    representation = transformers(patches_embed)

    y = GlobalAveragePooling1D()(representation)

    y =  MLP(projection_dim, num_classes, 0.2)(y) if num_classes > 0 else Lambda(lambda x: x)(y)

    model = Model(inputs=input, outputs=y)
    model.patchExtractor = patchExtractor
    model.patchEncoder = patchEncoder
    model.transformers = transformers
   
    return model


def get_last_selfattention(model, x, training=False):
    #x = model.input(x)
    patches = model.patchExtractor(x)
    #patches = model.reshape(patches)
    patches_embed = model.patchEncoder(patches)

    for i, blk in enumerate(model.transformers.blocks):
        if i < len(model.transformers.blocks) - 1:
            y = blk(patches_embed, training=training)
        else:
            return blk(patches_embed, training=training, return_attention=True)

def get_intermediate_attention_scores(model, x, n=1, training=False):

    patches = model.patchExtractor(x)
    #patches = model.reshape(patches)
    patches_embed = model.patchEncoder(patches)

    output = []
    attention_scores_list = []

    for i, blk in enumerate(model.transformers.blocks):
        y, attention_scores = blk(patches_embed, training=training, return_attention=True)

        if len(model.transformers.blocks) - i <= n:
            attention_scores_list.append(attention_scores)
            output.append(model.transformers.norm(y))

    return output, attention_scores_list


class ContrastiveModel(keras.Model):

    def __init__(self, encoder, projection_head_width, temperature=0.1, **kwargs):
        super(ContrastiveModel, self).__init__(**kwargs)

        self.temperature = temperature
        self.encoder = encoder
        self.projection_head_width = projection_head_width
       

        # Non-linear MLP as projection head
        self.projection_head = keras.Sequential(
            [
                keras.Input(shape=(self.projection_head_width,)),
                layers.BatchNormalization(),
                layers.Dense(self.projection_head_width, activation=keras.activations.selu),
                layers.BatchNormalization(),
                layers.Dense(self.projection_head_width, activation=None),
            ],
            name="projection_head",
        )

        print("Encoder - model weights: ", self.encoder.count_params())
        self.encoder.summary()

        print("Contrastive Projection Head - model weights: ", self.projection_head.count_params())
        self.projection_head.summary()
    
    def build(self, input_shape):
        #print(input_shape)
        #print(input_shape[0])
        """Ensures that the model is properly built with input shape"""
        self.encoder.build(input_shape=(input_shape[0]))  # Build encoder
        self.projection_head.build(input_shape=(None, self.projection_head_width))  # Build projection head
        super().build(input_shape)  # Call parent build

    def compile(self, contrastive_optimizer, **kwargs):
        super().compile(**kwargs)

        self.contrastive_optimizer = contrastive_optimizer

        self.contrastive_loss_tracker = keras.metrics.Mean(name="c_loss")
        self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy(
            name="c_acc"
        )
    
    @property
    def metrics(self):
        return [
            self.contrastive_loss_tracker,
            self.contrastive_accuracy,
        ]

    def contrastive_loss(self, projections_1, projections_2):
        # InfoNCE loss (information noise-contrastive estimation)
        # NT-Xent loss (normalized temperature-scaled cross entropy)

        # Cosine similarity: the dot product of the l2-normalized feature vectors
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)
        similarities = (
            tf.matmul(projections_1, projections_2, transpose_b=True) / self.temperature
        )

        # The similarity between the representations of two augmented views of the
        # same image should be higher than their similarity with other views
        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        self.contrastive_accuracy.update_state(contrastive_labels, similarities)
        self.contrastive_accuracy.update_state(
            contrastive_labels, tf.transpose(similarities)
        )

        # The temperature-scaled similarities are used as logits for cross-entropy
        # a symmetrized version of the loss is used here
        loss_1_2 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, similarities, from_logits=True
        )
        loss_2_1 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, tf.transpose(similarities), from_logits=True
        )
        return (loss_1_2 + loss_2_1) / 2
    
    def call(self, inputs, training=False):
        features_1 = self.encoder(inputs[0], training=training)
        features_2 = self.encoder(inputs[1], training=training)
        projections_1 = self.projection_head(features_1, training=training)
        projections_2 = self.projection_head(features_2, training=training)
        return projections_1, projections_2

    def get_config(self):
        config = super().get_config()
        config.update({
            "projection_head_width": self.projection_head_width,
            "temperature": self.temperature,
            "encoder_config": self.encoder.get_config(),
        })
        return config
    
    @classmethod
    def from_config(cls, config, custom_objects=None):
        """ Restore the encoder from its config"""
        encoder_config = config.pop("encoder_config")
        encoder = keras.Model.from_config(encoder_config, custom_objects=custom_objects)  # Restore encoder model
        return cls(encoder=encoder, **config)
    
    @tf.function
    def train_step(self, data):
        

        with tf.GradientTape() as tape:
            features_1 = self.encoder(data[0], training=True)
            features_2 = self.encoder(data[1], training=True)
            
            projections_1 = self.projection_head(features_1, training=True)
            projections_2 = self.projection_head(features_2, training=True)
            
            contrastive_loss = self.contrastive_loss(projections_1, projections_2)

        # Compute gradients
        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights + self.projection_head.trainable_weights,
        )

        # Update weights
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )

        # Update metrics
        self.contrastive_loss_tracker.update_state(contrastive_loss)
        return {m.name: m.result() for m in self.metrics}




if __name__ == "__main__":


    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a Vision Transformer contrastive model on GCP.")

    parser.add_argument("--input_dir", type=str, default="gs://dataset-dcts-fm/TFRecords", help="The TFRecords directory.") # Esse caminho vai funcionar?
    parser.add_argument("--output_dir", type=str, default="gs://dataset-dcts-fm/output-tpu", help="The directory to save the model and checkpoint.")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a saved model to resume training.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for training.")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for training.")
    parser.add_argument("--patch_size", type=int, default=8, help="Size of the patch the image is divided.")
    parser.add_argument("--num_patches", type=int, default=784, help="Number of patches resulted from the image.")
    parser.add_argument("--num_blocks", type=int, default=12, help="Number of transformer blocks.")
    parser.add_argument("--num_heads", type=int, default=6, help="Number of attention heads.")
    parser.add_argument("--projection_dim", type=int, default=384, help="Dimension of projection layers.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for contrastive loss.")
    parser.add_argument("--tpu", type=str, default=None, help="TPU name to be allocated")

    args = parser.parse_args()
    total_files = 1281167

    print("Arguments:", args._get_args())
    try:
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(args.tpu)
        tf.config.experimental_connect_to_cluster(cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        strategy = tf.distribute.TPUStrategy(cluster_resolver)
        print("✅ TPU detectada e inicializada corretamente.")
        print(f"TensorFlow can access {len(tf.config.list_logical_devices('TPU'))} TPU cores")
    except Exception as e:
        print(f"⚠️ Erro ao conectar à TPU: {e}")
        print("Executando em CPU/GPU em vez de TPU...")
        # strategy = tf.distribute.get_strategy()
        raise e


    try:
        # os.makedirs(args.output_dir, exist_ok=True)
        # os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        # os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
        # os.makedirs(os.path.join(args.output_dir, "model_save"), exist_ok=True)
        if not tf.io.gfile.exists(args.output_dir):
            tf.io.gfile.makedirs(args.output_dir)
            
        if not tf.io.gfile.exists(os.path.join(args.output_dir, "checkpoints")):
            tf.io.gfile.makedirs(os.path.join(args.output_dir, "checkpoints"))
        else:
            print("gfile does not exist!")
            
        if not tf.io.gfile.exists(os.path.join(args.output_dir, "logs")):
            tf.io.gfile.makedirs(os.path.join(args.output_dir, "logs"))
        
        if not tf.io.gfile.exists(os.path.join(args.output_dir, "model_save")):
            tf.io.gfile.makedirs(os.path.join(args.output_dir, "model_save"))


        checkpoint_path = os.path.join(args.output_dir, "checkpoints/cp_contrastive-vit.keras")
        tensorboard_log = os.path.join(args.output_dir, "logs")
        # Load dataset

        training_dataset = load_compressed_tfrecord_dataset(args.input_dir, batch_size=args.batch_size)

        with strategy.scope():
            contrastive_optimizer=Adam(learning_rate=args.learning_rate)

            # Load or create model
            if args.resume_from and tf.io.gfile.exists(args.resume_from):
                print(f"Loading model from {args.resume_from}...")
                custom_objects = {
                    "PatchExtractor": PatchExtractor,
                    "PatchEncoder": PatchEncoder,
                    "MLP": MLP,
                    "Block": Block,
                    "TransformerEncoder": TransformerEncoder,
                    "ContrastiveModel": ContrastiveModel,
                }
                contrastive_model = load_model(args.resume_from, custom_objects=custom_objects, compile=False)
                contrastive_model.compile(contrastive_optimizer=contrastive_optimizer)
                latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
                if latest_checkpoint:
                    contrastive_model.load_weights(latest_checkpoint)
                    print(f"Loaded weights from {latest_checkpoint}")
            else:
                print("Creating a new model...")
                input_shape = (224, 224, 3)
                encoder = create_vit_model(input_shape, args.num_patches, args.patch_size, args.projection_dim, args.num_blocks, args.num_heads)
                contrastive_model = ContrastiveModel(encoder, args.projection_dim, args.temperature)
                contrastive_model.compile(contrastive_optimizer=contrastive_optimizer)
                contrastive_model.build(input_shape=[(None, 224, 224, 3), (None, 224, 224, 3)])

            

            cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                                        mode="max",
                                                        monitor="c_acc",
                                                        save_best_only=True,
                                                        #save_weights_only=True,
                                                        verbose=1)

            # Early stopping
            es_callback = tf.keras.callbacks.EarlyStopping(monitor='c_loss',patience=7,
                                                            mode='min',
                                                            min_delta=1e-4,
                                                            restore_best_weights=False,
                                                            start_from_epoch=1,
                                                            verbose=1)
            
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log, histogram_freq=1)

            # Train model
            history = contrastive_model.fit(
                training_dataset,
                steps_per_epoch= total_files//args.batch_size,
                epochs=50,
                callbacks=[cp_callback, es_callback, tensorboard_callback], 
                verbose=1
            )

            # Save history
            with open(os.path.join(args.output_dir, "training_history.pkl"), "wb") as f:
                pickle.dump(history.history, f)

            # Save final model
            model_save_path = os.path.join(args.output_dir, "model_save/contrastive_model.keras")
            contrastive_model.save(model_save_path)
            print(f"Model saved at {model_save_path}")

    except Exception as e:
        print(f"⚠️ Erro: {e}")
