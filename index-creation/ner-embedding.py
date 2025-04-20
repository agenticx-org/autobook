import torch
from typing import List, Dict, Union, Optional
from gliner import GLiNER, GLiNERConfig
from transformers import AutoTokenizer
import warnings

# --- Monkey Patch Definition ---
# This is the corrected version of the function causing the error.
def patched_extract_elements(sequence, indices):
    """
    Gathers elements from sequence based on indices, handling out-of-bounds.
    This is a patched version to replace the original gliner.modeling.span_rep.extract_elements.

    Args:
        sequence (torch.Tensor): Shape [B, L, D]. The source tensor.
        indices (torch.Tensor): Shape [B, K]. Indices to gather along dimension 1.

    Returns:
        torch.Tensor: Shape [B, K, D]. Gathered elements, with zeros for out-of-bounds indices.
    """
    B, L, D = sequence.shape
    K = indices.shape[1] # Number of spans/indices per batch item

    # Handle potential edge case where sequence length L is 0
    if L == 0:
        return torch.zeros(B, K, D, dtype=sequence.dtype, device=sequence.device)

    # Create a mask for valid indices (0 <= index < L)
    valid_mask = (indices >= 0) & (indices < L) # Shape [B, K]

    # Clamp indices to avoid gather error, but use mask to zero out invalid ones later
    indices_clamped = torch.clamp(indices, 0, L - 1) # Shape [B, K]

    # Expand clamped indices for gather
    expanded_indices = indices_clamped.unsqueeze(2).expand(-1, -1, D) # Shape [B, K, D]

    # Perform gather using clamped indices
    extracted_elements = torch.gather(sequence, 1, expanded_indices) # Shape [B, K, D]

    # Zero out elements corresponding to originally invalid indices
    # Ensure mask is compatible type for multiplication
    extracted_elements = extracted_elements * valid_mask.unsqueeze(2).type_as(extracted_elements)

    return extracted_elements

# --- Apply the Monkey Patch ---
# Import the specific module where the function resides
import gliner.modeling.span_rep

# Overwrite the original function with our patched version
# This must happen *before* the model tries to use the function internally
gliner.modeling.span_rep.extract_elements = patched_extract_elements
print("--- Monkey patch applied to gliner.modeling.span_rep.extract_elements ---")
# --- End Monkey Patch ---


@torch.no_grad()
def get_ner_with_embeddings(
    model: GLiNER,
    texts: Union[str, List[str]],
    labels: List[str],
    threshold: float = 0.5,
    batch_size: int = 8,
    flat_ner=True, # Consistent with predict_entities default
    multi_label=False # Consistent with predict_entities default
) -> List[List[Dict]]:
    """
    Performs Named Entity Recognition (NER) using a GLiNER model and returns
    entity details along with their span embeddings. Includes safety checks
    for index boundaries. Uses the monkey-patched extract_elements internally.

    Args:
        model (GLiNER): An initialized GLiNER model instance.
        texts (Union[str, List[str]]): A single text string or a list of texts.
        labels (List[str]): A list of entity labels to predict.
        threshold (float, optional): Confidence threshold for predictions. Defaults to 0.5.
        batch_size (int, optional): Batch size for processing. Defaults to 8.
        flat_ner (bool, optional): Whether to enforce non-overlapping spans (except for multi-label). Defaults to True.
        multi_label (bool, optional): Whether to allow multiple labels per span. Defaults to False.


    Returns:
        List[List[Dict]]: A list of lists, where each inner list corresponds to a text
                         input and contains dictionaries for each predicted entity.
                         Each entity dictionary includes:
                         - 'start': Start character index.
                         - 'end': End character index.
                         - 'text': The extracted entity text.
                         - 'label': The predicted entity label.
                         - 'score': The prediction confidence score.
                         - 'embedding': The span embedding (torch.Tensor) for the entity, or None if indices were invalid.
    """
    if isinstance(texts, str):
        texts = [texts]

    if model.config.span_mode == "token_level":
         warnings.warn("Span embeddings are most relevant for span-based models (like markerV0). "
                       "This function might return less meaningful embeddings for token-level models.")
         # Embedding extraction logic below might need adjustment for token-level models

    model.eval()
    all_results = []
    device = model.device

    # Process texts in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        # 1. Prepare Model Inputs
        model_input, raw_batch = model.prepare_model_inputs(batch_texts, labels)

        # 2. Run Forward Pass (This will now use the patched extract_elements internally)
        try:
            outputs = model.model(**model_input)
        except Exception as e:
             # Catch potential errors during the patched forward pass as well
             print(f"\nError during model forward pass (batch starting at index {i}): {e}")
             # Append empty results for this batch and continue
             for _ in batch_texts:
                 all_results.append([])
             continue # Skip the rest of the loop for this batch

        words_embedding = outputs.words_embedding
        logits = outputs.logits

        if not isinstance(logits, torch.Tensor):
                logits = torch.from_numpy(logits).to(device)
        if not isinstance(words_embedding, torch.Tensor):
                words_embedding = torch.from_numpy(words_embedding).to(device)

        # 3. Calculate Span Representations (if applicable)
        span_idx = model_input.get("span_idx")
        span_rep = None # Default
        if "marker" in model.config.span_mode.lower(): # Check if span mode requires span_idx
            if span_idx is None:
                 # This shouldn't happen if prepare_model_inputs worked correctly for this mode
                 warnings.warn("Span mode 'marker*' requires 'span_idx', but it's missing. Cannot calculate embeddings.")
            else:
                 span_idx = span_idx.to(device)
                 # This calculation should now be safe because extract_elements inside is patched
                 span_rep = model.model.span_rep_layer(words_embedding, span_idx)
                 # Shape: (batch, seq_len, max_width, hidden_dim)
        elif model.config.span_mode != "token_level":
            # Handle other potential span modes that might need span_idx later
            if span_idx is None:
                 warnings.warn(f"Span mode '{model.config.span_mode}' might require 'span_idx', but it's missing.")
            else:
                 # Generic calculation if needed, though specific layer access might differ
                 span_idx = span_idx.to(device)
                 # Placeholder: actual calculation might vary for other span modes
                 try:
                     span_rep = model.model.span_rep_layer(words_embedding, span_idx)
                 except AttributeError:
                     warnings.warn(f"Could not find span_rep_layer for mode '{model.config.span_mode}'. Cannot calculate embeddings.")


        # 4. Decode Logits
        decoded_spans_batch = model.decoder.decode(
            raw_batch["tokens"],
            raw_batch["id_to_classes"],
            logits,
            flat_ner=flat_ner,
            threshold=threshold,
            multi_label=multi_label
        )

        # 5. Map Decoded Spans to Embeddings and Format Output
        for j, decoded_spans in enumerate(decoded_spans_batch):
            current_text = batch_texts[j]
            # Use try-except for potential issues getting maps for this batch item
            try:
                 start_token_map = raw_batch["all_start_token_idx_to_text_idx"][j]
                 end_token_map = raw_batch["all_end_token_idx_to_text_idx"][j]
                 num_original_tokens = len(start_token_map) # Length of the original token sequence for this item
                 if span_rep is not None:
                     span_rep_seq_len = span_rep.shape[1] # Padded length in span_rep tensor for the sequence dim
                     span_rep_max_width = span_rep.shape[2]
                 else:
                     span_rep_seq_len = 0
                     span_rep_max_width = 0

            except IndexError:
                 warnings.warn(f"Could not retrieve token maps for batch item {j+i}. Skipping entities for this text.")
                 all_results.append([]) # Add empty list for this text
                 continue


            entities_with_embeddings = []
            for start_token_idx, end_token_idx, ent_type, ent_score in decoded_spans:

                # --- Safety Check 1: Token indices vs Original Token List Length ---
                if start_token_idx >= num_original_tokens or end_token_idx >= num_original_tokens:
                    warnings.warn(
                        f"Decoder produced token indices ({start_token_idx}, {end_token_idx}) "
                        f"out of bounds for original token list length ({num_original_tokens}). Skipping span."
                    )
                    continue

                # Indices are valid for maps, get character indices
                start_char_idx = start_token_map[start_token_idx]
                end_char_idx = end_token_map[end_token_idx]

                span_embedding = None # Default to None

                if span_rep is not None: # Only calculate embedding if span_rep exists
                    # Calculate width index
                    width_index = end_token_idx - start_token_idx

                    # --- Safety Check 2: Indices vs Span Embedding Tensor Dimensions ---
                    # These checks primarily ensure the decoder didn't produce something unexpected
                    # The monkey patch handles the internal gather error.
                    if start_token_idx >= span_rep_seq_len:
                         warnings.warn(
                            f"Start token index {start_token_idx} is out of bounds for span_rep sequence length "
                            f"({span_rep_seq_len}). Skipping span embedding retrieval."
                        )
                    elif width_index < 0 or width_index >= span_rep_max_width:
                         warnings.warn(
                            f"Calculated width index {width_index} is out of bounds for span_rep max width "
                            f"({span_rep_max_width}). Skipping span embedding retrieval."
                         )
                    else:
                         # All checks passed, get the embedding
                         # This access should now be safe due to the monkey patch handling internal indices
                         span_embedding = span_rep[j, start_token_idx, width_index, :].clone().detach()


                entities_with_embeddings.append(
                    {
                        "start": start_char_idx,
                        "end": end_char_idx,
                        "text": current_text[start_char_idx:end_char_idx],
                        "label": ent_type,
                        "score": ent_score,
                        "embedding": span_embedding # Add the embedding (might be None)
                    }
                )
            all_results.append(entities_with_embeddings)

    return all_results

# --- Example Usage ---
if __name__ == "__main__":
    # Load a pre-trained GLiNER model (make sure it's a span-based one like markerV0)
    model_name = "urchade/gliner_small-v2.1" # Model known to use markerV0

    try:
        # Load the model AFTER the patch has been applied
        gliner_model = GLiNER.from_pretrained(model_name)
        print(f"\nLoaded model: {model_name}")
        print(f"Model span mode: {gliner_model.config.span_mode}")

        if "marker" not in gliner_model.config.span_mode.lower() and "token" not in gliner_model.config.span_mode.lower():
             print("\nWarning: The loaded model might not be ideal for extracting meaningful span embeddings "
                   f"as its span_mode is '{gliner_model.config.span_mode}'. Results might vary.")

        text1 = "John Doe visited Paris last week. He works at Acme Corp."
        # Add a text that might be closer to the problematic length if known
        text2 = "Apple Inc. is planning to release a new iPhone in September, sources say."
        texts = [text1, text2]
        labels = ["person", "location", "date", "organization", "product", "company"] # Added company

        # Call the function which will use the patched model
        results = get_ner_with_embeddings(gliner_model, texts, labels, threshold=0.5)

        print("\n--- Results ---")
        for i, text_result in enumerate(results):
            print(f"\nText {i+1}: '{texts[i]}'")
            if not text_result:
                print("  No entities found.")
                continue
            for entity in text_result:
                print(f"  Entity: '{entity['text']}'")
                print(f"    Label: {entity['label']}")
                print(f"    Score: {entity['score']:.4f}")
                print(f"    Span: ({entity['start']}, {entity['end']})")
                if entity['embedding'] is not None:
                    print(f"    Embedding Shape: {entity['embedding'].shape}")
                    # print(f"    Embedding Sample: {entity['embedding'][:5]}...") # Uncomment to see embedding values
                else:
                    print("    Embedding: None")

    except ImportError:
        print("\nPlease ensure 'gliner', 'torch', and 'transformers' are installed.")
        print("`pip install gliner torch transformers`")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure the 'gliner' library and its dependencies are installed (`pip install gliner`)")
        print("Also check if the model ID is correct and accessible, and that the monkey patch was applied successfully.")
        # You might want to add more detailed error logging here if needed
        import traceback
        traceback.print_exc()