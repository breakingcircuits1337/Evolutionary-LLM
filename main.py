#
# A conceptual demonstration of evolving the weights of a complex, multi-modal
# Transformer LLM using the evolutionary principles from agi_simulation.py.
#
# We are replacing a standard gradient-based optimizer (like Adam) with a
# population-based, gradient-free evolutionary search algorithm.
#

import tensorflow as tf
from transformers import BertTokenizer, TFAutoModel
import numpy as np
import random
import copy
import faiss  # Make sure faiss-cpu or faiss-gpu is installed

# --- LLM ARCHITECTURE COMPONENTS (from user's script) ---
# Note: Removed excessive print statements for clarity during evolution.

class HParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def gelu(x):
    return 0.5 * x * (1 + tf.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, n_embd, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.c_attn = tf.keras.layers.Dense(3 * n_embd)
        self.c_proj = tf.keras.layers.Dense(n_embd)

    def split_heads(self, x):
        batch_size, seq_length, _ = tf.unstack(tf.shape(x))
        return tf.reshape(x, (batch_size, seq_length, self.n_head, self.n_embd // self.n_head))

    def merge_heads(self, x):
        batch_size, seq_length, _, d_head = tf.unstack(tf.shape(x))
        return tf.reshape(x, (batch_size, seq_length, self.n_head * d_head))

    def call(self, x, past=None):
        c = self.c_attn(x)
        q, k, v = tf.split(c, 3, axis=-1)
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        attention_scores = tf.einsum('bshd,bthd->bst', q, k) / tf.math.sqrt(tf.cast(v.shape[-1], tf.float32))
        attention_weights = tf.nn.softmax(attention_scores)
        attention_output = tf.einsum('bst,bthd->bshd', attention_weights, v)
        attention_output = self.merge_heads(attention_output)
        a = self.c_proj(attention_output)
        return a, None # Simplified for non-causal attention

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, n_embd, n_head):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(n_embd, n_head)
        self.ln_1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(4 * n_embd, activation=gelu),
            tf.keras.layers.Dense(n_embd)
        ])
        self.ln_2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    def call(self, x, past=None):
        a, _ = self.attn(self.ln_1(x), past=past)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, None

class FAISSRetriever:
    def __init__(self, knowledge_base_vectors, knowledge_base_text, dim=768, num_results=3):
        self.index = faiss.IndexFlatL2(dim)
        self.knowledge_base_text = knowledge_base_text
        self.num_results = num_results
        if knowledge_base_vectors.shape[0] > 0:
            self.index.add(knowledge_base_vectors)

    def retrieve(self, query_vector):
        if self.index.ntotal == 0:
            return ""
        query_vector = query_vector.numpy() if tf.is_tensor(query_vector) else query_vector
        query_vector = np.expand_dims(query_vector, axis=0) if len(query_vector.shape) == 1 else query_vector
        _, indices = self.index.search(query_vector.astype('float32'), self.num_results)
        return " ".join([self.knowledge_base_text[i] for i in indices[0]])

class MultiModalTransformer(tf.keras.Model):
    def __init__(self, hparams, knowledge_base_vectors, knowledge_base_text, tokenizer, n_hash=1024, n_quant=256):
        super(MultiModalTransformer, self).__init__()
        self.hparams = hparams
        self.tokenizer = tokenizer
        self.wte = tf.keras.layers.Embedding(hparams.n_vocab, hparams.n_embd)
        self.wpe = tf.keras.layers.Embedding(hparams.n_ctx, hparams.n_embd)
        self.hash_layer = tf.keras.layers.Dense(n_hash, activation='relu')
        self.quant_layer = tf.keras.layers.Dense(n_quant, activation='relu')
        self.h = [TransformerBlock(n_quant, hparams.n_head) for _ in range(hparams.n_layer)]
        self.ln_f = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.fc = tf.keras.layers.Dense(hparams.n_vocab, use_bias=False)
        self.retriever = FAISSRetriever(knowledge_base_vectors, knowledge_base_text, dim=hparams.n_embd)
        self.query_encoder = tf.keras.layers.Dense(hparams.n_embd)

    def call(self, inputs, task='text_generation'):
        # Simplified for a single task (text generation) for evolutionary clarity
        x = self.wte(inputs)
        query_embedding = tf.reduce_mean(x, axis=1)
        encoded_query = self.query_encoder(query_embedding)
        retrieved_text = self.retriever.retrieve(encoded_query)
        retrieved_tokens = self.tokenizer(retrieved_text, return_tensors="tf", padding='max_length', max_length=50, truncation=True).input_ids
        retrieved_embeds = self.wte(retrieved_tokens)
        x = tf.concat([retrieved_embeds, x], axis=1)
        
        seq_len = tf.shape(x)[1]
        if seq_len > self.hparams.n_ctx:
            x = x[:, :self.hparams.n_ctx, :]
            seq_len = self.hparams.n_ctx

        position = tf.range(0, seq_len, dtype=tf.int32)[tf.newaxis, :]
        x = x + self.wpe(position)
        x = self.hash_layer(x)
        x = self.quant_layer(x)
        for layer in self.h:
            x, _ = layer(x)
        x = self.ln_f(x)
        output = self.fc(x)
        return output

# --- EVOLUTIONARY ENGINE (adapted from agi_simulation.py) ---

class EvolutionaryOptimizer:
    """
    Manages a population of MultiModalTransformer models and orchestrates the evolutionary process.
    """
    def __init__(self, population_size, mutation_rate, hparams, kb_vectors, kb_text, tokenizer, elitism=0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elitism_count = int(population_size * elitism)
        self.generation = 0
        
        # --- FIX: Store model creation parameters for later use ---
        self.hparams = hparams
        self.kb_vectors = kb_vectors
        self.kb_text = kb_text
        self.tokenizer = tokenizer
        
        print("Initializing population of LLMs...")
        self.population = [self._create_new_model() for _ in range(population_size)]
        print("Population initialized.")

    def _create_new_model(self):
        """Helper function to create a new model instance and build it."""
        model = MultiModalTransformer(self.hparams, self.kb_vectors, self.kb_text, self.tokenizer)
        # Build the model by calling it once with a dummy input
        dummy_input = tf.zeros((1, 10), dtype=tf.int32)
        model(dummy_input)
        return model

    def calculate_fitness(self, model, problem_data):
        """Calculates fitness based on the model's performance on a task."""
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        total_loss = 0
        for data in problem_data:
            inputs = data['inputs']
            targets = data['targets']
            predictions = model(inputs)
            
            # --- FIX FOR SHAPE MISMATCH ---
            # The predictions are for the sequence [retrieved_context + prompt].
            # The targets are for the sequence [prompt + completion].
            # We must align the slices of these tensors to calculate a meaningful loss.

            # The retrieved context is padded to 50 tokens. The predictions for the
            # original prompt start after this context.
            retrieved_context_len = 50
            prompt_len = inputs.shape[1]

            # Slice predictions to get the part corresponding to the prompt.
            # Shape: (batch, prompt_len, vocab_size)
            pred_slice = predictions[:, retrieved_context_len:(retrieved_context_len + prompt_len), :]

            # Slice targets to match the length of the prediction slice.
            # We compare the model's output for the prompt against the correct tokens for the prompt.
            # Shape: (batch, prompt_len)
            target_slice = targets[:, :prompt_len]
            
            # Ensure sequence lengths match before calculating loss. This prevents errors
            # if tokenization results in unexpected lengths.
            if pred_slice.shape[1] == target_slice.shape[1] and pred_slice.shape[1] > 0:
                loss = loss_fn(target_slice, pred_slice)
                total_loss += tf.reduce_mean(loss)
            else:
                # If shapes can't be aligned, this model is invalid. Assign a very
                # high loss to give it a very low fitness score.
                total_loss += 1e9

        # Fitness is the inverse of loss. Add a small number to avoid division by zero.
        return 1 / (total_loss + 1e-8)

    def selection(self, fitness_scores):
        """Selects parents for the next generation based on fitness."""
        fitness_sum = sum(fitness_scores)
        selection_probs = [score / fitness_sum for score in fitness_scores]
        
        selected_parents = random.choices(self.population, weights=selection_probs, k=self.population_size - self.elitism_count)
        return selected_parents

    def crossover(self, parent1, parent2):
        """Combines the weights of two parent models to create a child."""
        # --- FIX: Manually create a new model instead of deepcopy ---
        child = self._create_new_model()
        
        p1_weights = parent1.get_weights()
        p2_weights = parent2.get_weights()
        child_weights = []

        for w1, w2 in zip(p1_weights, p2_weights):
            # Uniform crossover: for each layer, randomly pick which parent's weights to use
            if random.random() < 0.5:
                child_weights.append(w1)
            else:
                child_weights.append(w2)
        
        child.set_weights(child_weights)
        return child

    def mutate(self, model):
        """Introduces small, random changes into a model's weights."""
        weights = model.get_weights()
        mutated_weights = []
        for w in weights:
            if random.random() < self.mutation_rate:
                noise = tf.random.normal(shape=w.shape, stddev=0.01)
                mutated_weights.append(w + noise.numpy())
            else:
                mutated_weights.append(w)
        model.set_weights(mutated_weights)
        return model

    def evolve_generation(self, problem_data):
        """Orchestrates one full cycle of evolution."""
        fitness_scores = [self.calculate_fitness(model, problem_data) for model in self.population]

        sorted_population = [x for _, x in sorted(zip(fitness_scores, self.population), key=lambda pair: pair[0], reverse=True)]
        
        parents = self.selection(fitness_scores)
        
        # --- FIX: Manually create copies of elites instead of deepcopy ---
        next_generation = []
        for elite_model in sorted_population[:self.elitism_count]:
            new_elite = self._create_new_model()
            new_elite.set_weights(elite_model.get_weights())
            next_generation.append(new_elite)

        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                parent1, parent2 = parents[i], parents[i+1]
                child = self.crossover(parent1, parent2)
                mutated_child = self.mutate(child)
                next_generation.append(mutated_child)
        
        # Ensure population size is maintained
        while len(next_generation) < self.population_size:
            p1, p2 = random.choices(parents, k=2)
            child = self.crossover(p1, p2)
            next_generation.append(self.mutate(child))
        
        self.population = next_generation[:self.population_size]
        self.generation += 1
        return max(fitness_scores)

# --- Main Execution ---
if __name__ == '__main__':
    # --- 1. HYPERPARAMETERS AND SETUP ---
    # WARNING: This is EXTREMELY computationally expensive.
    # These parameters are for demonstration only.
    POPULATION_SIZE = 10  # In reality, this would be much larger
    MUTATION_RATE = 0.02
    GENERATIONS = 5 # In reality, this would be thousands
    
    hparams = HParams(
        n_vocab=30522, n_ctx=128, n_embd=128, n_head=4, n_layer=2
    )
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # --- 2. KNOWLEDGE BASE SETUP ---
    knowledge_base = ["The Eiffel Tower is in Paris.", "Photosynthesis is a plant process."]
    knowledge_base_vectors = np.random.rand(len(knowledge_base), hparams.n_embd).astype('float32')

    # --- 3. PROBLEM DEFINITION (the "environment" for evolution) ---
    # We want the LLM to learn to complete a sentence.
    prompt = "The Eiffel Tower is in"
    target_sentence = "The Eiffel Tower is in Paris."
    
    input_tokens = tokenizer(prompt, return_tensors="tf").input_ids
    target_tokens = tokenizer(target_sentence, return_tensors="tf").input_ids
    
    problem_data = [{'inputs': input_tokens, 'targets': target_tokens}]

    # --- 4. EVOLUTIONARY OPTIMIZER SETUP ---
    ecosystem = EvolutionaryOptimizer(
        POPULATION_SIZE, MUTATION_RATE, hparams, knowledge_base_vectors, knowledge_base, tokenizer
    )
    
    # --- 5. THE MAIN EVOLUTIONARY LOOP ---
    print("\n--- Starting Evolution of LLM Population ---")
    for i in range(GENERATIONS):
        best_fitness = ecosystem.evolve_generation(problem_data)
        print(f"Generation: {i + 1:2d} | Best Fitness (1/loss): {best_fitness:.4f}")
    
    print("\n--- Evolution Finished ---")

    # --- 6. Find and Test the Best Evolved Model ---
    fitness_scores = [ecosystem.calculate_fitness(model, problem_data) for model in ecosystem.population]
    best_model = ecosystem.population[np.argmax(fitness_scores)]

    print("\n--- Testing the Best Evolved LLM ---")
    input_text = "The Eiffel Tower is in"
    input_ids = tokenizer(input_text, return_tensors="tf").input_ids
    output_logits = best_model(input_ids)
    predicted_token_id = tf.argmax(output_logits[0, -1, :]).numpy()
    predicted_token = tokenizer.decode([predicted_token_id])
    
    print(f"Prompt: '{input_text}'")
    print(f"Predicted next token: '{predicted_token}'")



