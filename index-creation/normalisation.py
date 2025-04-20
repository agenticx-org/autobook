import torch
import torch.nn.functional as F
from typing import List, Dict, Union, Optional, Tuple
from gliner import GLiNER, GLiNERConfig
from transformers import AutoTokenizer
import warnings
import re # For basic cleaning
from rapidfuzz import fuzz # For lexical similarity

# --- Monkey Patch Definition (Keep this from the previous step) ---
def patched_extract_elements(sequence, indices):
    B, L, D = sequence.shape
    K = indices.shape[1]
    if L == 0: return torch.zeros(B, K, D, dtype=sequence.dtype, device=sequence.device)
    valid_mask = (indices >= 0) & (indices < L)
    indices_clamped = torch.clamp(indices, 0, L - 1)
    expanded_indices = indices_clamped.unsqueeze(2).expand(-1, -1, D)
    extracted_elements = torch.gather(sequence, 1, expanded_indices)
    extracted_elements = extracted_elements * valid_mask.unsqueeze(2).type_as(extracted_elements)
    return extracted_elements

import gliner.modeling.span_rep
gliner.modeling.span_rep.extract_elements = patched_extract_elements
print("--- Monkey patch applied to gliner.modeling.span_rep.extract_elements ---")
# --- End Monkey Patch ---


# --- NER Function (Keep this from the previous step) ---
@torch.no_grad()
def get_ner_with_embeddings(
    model: GLiNER,
    texts: Union[str, List[str]],
    labels: List[str],
    threshold: float = 0.5,
    batch_size: int = 8,
    flat_ner=True,
    multi_label=False
) -> List[List[Dict]]:
    # (Implementation from the previous answer remains the same)
    # ... (rest of the get_ner_with_embeddings function code) ...
    if isinstance(texts, str):
        texts = [texts]

    if model.config.span_mode == "token_level":
         warnings.warn("Span embeddings are most relevant for span-based models (like markerV0). "
                       "This function might return less meaningful embeddings for token-level models.")

    model.eval()
    all_results = []
    device = model.device

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        model_input, raw_batch = model.prepare_model_inputs(batch_texts, labels)

        try:
            outputs = model.model(**model_input)
        except Exception as e:
             print(f"\nError during model forward pass (batch starting at index {i}): {e}")
             for _ in batch_texts: all_results.append([])
             continue

        words_embedding = outputs.words_embedding
        logits = outputs.logits

        if not isinstance(logits, torch.Tensor): logits = torch.from_numpy(logits).to(device)
        if not isinstance(words_embedding, torch.Tensor): words_embedding = torch.from_numpy(words_embedding).to(device)

        span_idx = model_input.get("span_idx")
        span_rep = None
        if "marker" in model.config.span_mode.lower():
            if span_idx is None: warnings.warn("Span mode 'marker*' requires 'span_idx', but it's missing. Cannot calculate embeddings.")
            else:
                 span_idx = span_idx.to(device)
                 span_rep = model.model.span_rep_layer(words_embedding, span_idx)
        elif model.config.span_mode != "token_level":
             if span_idx is None: warnings.warn(f"Span mode '{model.config.span_mode}' might require 'span_idx', but it's missing.")
             else:
                try:
                     span_idx = span_idx.to(device)
                     span_rep = model.model.span_rep_layer(words_embedding, span_idx)
                except AttributeError: warnings.warn(f"Could not find span_rep_layer for mode '{model.config.span_mode}'. Cannot calculate embeddings.")

        decoded_spans_batch = model.decoder.decode(
            raw_batch["tokens"], raw_batch["id_to_classes"], logits,
            flat_ner=flat_ner, threshold=threshold, multi_label=multi_label
        )

        for j, decoded_spans in enumerate(decoded_spans_batch):
            current_text = batch_texts[j]
            try:
                 start_token_map = raw_batch["all_start_token_idx_to_text_idx"][j]
                 end_token_map = raw_batch["all_end_token_idx_to_text_idx"][j]
                 num_original_tokens = len(start_token_map)
                 if span_rep is not None:
                     span_rep_seq_len, span_rep_max_width = span_rep.shape[1], span_rep.shape[2]
                 else: span_rep_seq_len, span_rep_max_width = 0, 0
            except IndexError:
                 warnings.warn(f"Could not retrieve token maps for batch item {j+i}. Skipping entities for this text.")
                 all_results.append([])
                 continue

            entities_with_embeddings = []
            for start_token_idx, end_token_idx, ent_type, ent_score in decoded_spans:
                if start_token_idx >= num_original_tokens or end_token_idx >= num_original_tokens:
                    warnings.warn(f"Decoder produced token indices ({start_token_idx}, {end_token_idx}) out of bounds for original token list length ({num_original_tokens}). Skipping span.")
                    continue

                start_char_idx = start_token_map[start_token_idx]
                end_char_idx = end_token_map[end_token_idx]
                span_embedding = None

                if span_rep is not None:
                    width_index = end_token_idx - start_token_idx
                    if start_token_idx >= span_rep_seq_len: warnings.warn(f"Start token index {start_token_idx} out of bounds for span_rep sequence length ({span_rep_seq_len}). Skipping span embedding retrieval.")
                    elif width_index < 0 or width_index >= span_rep_max_width: warnings.warn(f"Calculated width index {width_index} out of bounds for span_rep max width ({span_rep_max_width}). Skipping span embedding retrieval.")
                    else: span_embedding = span_rep[j, start_token_idx, width_index, :].clone().detach()

                entities_with_embeddings.append({
                    "start": start_char_idx, "end": end_char_idx,
                    "text": current_text[start_char_idx:end_char_idx],
                    "label": ent_type, "score": ent_score, "embedding": span_embedding
                })
            all_results.append(entities_with_embeddings)
    return all_results


# --- Normalization Function (Lexical First) ---

def clean_text_for_map(text: str) -> str:
    """Basic cleaning: lowercase and remove specified punctuation, normalize whitespace."""
    text = text.lower()
    # Remove common punctuation, keeping internal hyphens for cases like 'state-of-the-art'
    # Allow numbers as well
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
    return text

def get_canonical_name_part(canonical_id: str) -> Optional[str]:
     """Extracts the name part from a canonical ID (e.g., 'ORG:Tesla' -> 'Tesla')."""
     parts = canonical_id.split(':', 1)
     if len(parts) == 2:
         return parts[1]
     return None # Return None if format is incorrect

def normalize_entities_lexical_first(
    ner_results: List[List[Dict]],
    normalization_map: Dict[str, str],
    canonical_embeddings: Dict[str, torch.Tensor],
    lexical_threshold: float = 80.0, # Fuzzy matching threshold (0-100)
    embedding_threshold: float = 0.80, # Cosine similarity threshold
    use_embedding_confirmation: bool = True, # Use embedding even for single lexical match?
    device: str = 'cpu',
    debug: bool = False
) -> Tuple[List[List[Dict]], Dict[str, str], Dict[str, torch.Tensor]]:
    """
    Normalizes entities using map lookup, then lexical similarity,
    and finally embedding similarity as confirmation or tie-breaker.

    Args:
        ner_results: Output from get_ner_with_embeddings.
        normalization_map: Dict mapping cleaned aliases to canonical IDs (LABEL:Name). Updated.
        canonical_embeddings: Dict mapping canonical IDs to embeddings. Updated.
        lexical_threshold: Threshold (0-100) for rapidfuzz ratio matching.
        embedding_threshold: Cosine similarity threshold for embedding confirmation/tie-breaking.
        use_embedding_confirmation: If True, require embedding match even if only one lexical candidate found.
        device: Torch device for embedding calculations.
        debug: If True, print verbose matching information.

    Returns:
        Tuple: (updated_ner_results, updated_normalization_map, updated_canonical_embeddings)
    """
    updated_ner_results = []
    current_normalization_map = normalization_map.copy()
    current_canonical_embeddings = canonical_embeddings.copy()
    # Create a reverse map for quick lookup of canonical names (cleaned) to ID
    # This assumes one primary cleaned name per ID for simplicity here, could be more complex
    canonical_name_map = {
        clean_text_for_map(name): cid
        for cid in current_canonical_embeddings.keys() # Use embedding keys as source of truth for names
        if (name := get_canonical_name_part(cid)) is not None
    }
    # Add names derived from the normalization map as well, potentially overwriting
    for alias, cid in current_normalization_map.items():
         if (name := get_canonical_name_part(cid)) is not None:
              cleaned_name = clean_text_for_map(name)
              if cleaned_name not in canonical_name_map: # Avoid overwriting if already from embeddings
                   canonical_name_map[cleaned_name] = cid


    for text_entities in ner_results:
        processed_text_entities = []
        for entity in text_entities:
            entity_copy = entity.copy()
            assigned_canonical_id = None
            entity_text_cleaned = clean_text_for_map(entity_copy['text'])
            entity_label = entity_copy['label'].upper()
            entity_embedding = entity_copy.get('embedding')

            if not entity_text_cleaned: # Skip empty strings after cleaning
                entity_copy['canonical_id'] = f"{entity_label}:<EMPTY>" # Assign a special ID
                processed_text_entities.append(entity_copy)
                continue

            # --- Step 1: Check the normalization map ---
            if entity_text_cleaned in current_normalization_map:
                assigned_canonical_id = current_normalization_map[entity_text_cleaned]
                if debug: print(f"  Match for '{entity_copy['text']}': Found alias '{entity_text_cleaned}' -> '{assigned_canonical_id}' in map.")

            # --- Step 2: Lexical Similarity Search (if not in map) ---
            if assigned_canonical_id is None:
                lexical_candidates = [] # List of (canonical_id, lexical_score)
                if debug: print(f"  Match for '{entity_copy['text']}': Alias '{entity_text_cleaned}' not in map. Searching lexically...")

                # Iterate through known canonical IDs (use embedding keys + map keys for comprehensive search)
                known_canonical_ids = set(current_canonical_embeddings.keys()) | set(current_normalization_map.values())

                for known_cid in known_canonical_ids:
                    try:
                        known_label = known_cid.split(':')[0].upper()
                        known_name = get_canonical_name_part(known_cid)
                    except:
                         if debug: warnings.warn(f"Skipping invalid canonical ID format: {known_cid}")
                         continue

                    if known_label == entity_label and known_name:
                        known_name_cleaned = clean_text_for_map(known_name)
                        if not known_name_cleaned: continue # Skip empty canonical names

                        # Lexical Checks:
                        # 1. Substring/Superstring (simple version)
                        is_substring = entity_text_cleaned in known_name_cleaned
                        is_superstring = known_name_cleaned in entity_text_cleaned

                        # 2. Fuzzy Match Ratio (catches typos, minor variations)
                        fuzzy_score = fuzz.ratio(entity_text_cleaned, known_name_cleaned)

                        # Combine checks - require high fuzzy score OR substring/superstring
                        if is_substring or is_superstring or fuzzy_score >= lexical_threshold:
                            # Store the highest score found for this candidate ID
                            # Use fuzzy score primarily, maybe boost if substring/superstring? (Keep simple for now)
                            score_to_store = fuzzy_score # Or max(fuzzy_score, 95 if is_substring or is_superstring else 0)
                            lexical_candidates.append((known_cid, score_to_store))
                            if debug: print(f"    Lexical candidate: '{known_cid}' (Name: '{known_name_cleaned}') - Score: {score_to_store:.1f} (Sub: {is_substring}, Super: {is_superstring}, Fuzzy: {fuzzy_score:.1f})")


                # Remove duplicate candidate IDs, keep the one with the highest score
                if lexical_candidates:
                    candidate_dict = {}
                    for cid, score in lexical_candidates:
                         candidate_dict[cid] = max(score, candidate_dict.get(cid, -1.0))
                    lexical_candidates = sorted(candidate_dict.items(), key=lambda item: item[1], reverse=True)
                    if debug: print(f"    Unique lexical candidates sorted: {lexical_candidates}")


                # --- Step 3: Embedding Confirmation/Tie-breaking ---
                if lexical_candidates:
                    best_candidate_cid = None
                    if entity_embedding is None:
                         # No embedding available for the new entity, just take the best lexical match
                         best_candidate_cid = lexical_candidates[0][0] # Highest lexical score
                         if debug: print(f"    Decision: No embedding for '{entity_copy['text']}'. Assigning best lexical match: '{best_candidate_cid}'")
                    else:
                         # Embedding available, compare against candidates
                         entity_embedding = entity_embedding.to(device)
                         best_embedding_match_cid = None
                         highest_embedding_similarity = -1.0

                         candidates_to_check = []
                         if len(lexical_candidates) == 1 and not use_embedding_confirmation:
                              # Option: Directly assign if only one lexical candidate and confirmation is off
                              # best_candidate_cid = lexical_candidates[0][0]
                              # if debug: print(f"    Decision: Only 1 lexical candidate ('{best_candidate_cid}') and confirmation off. Assigning.")
                              # Let's proceed with check anyway for consistency for now
                              candidates_to_check = lexical_candidates
                         else:
                              candidates_to_check = lexical_candidates

                         for candidate_cid, lex_score in candidates_to_check:
                              if candidate_cid in current_canonical_embeddings:
                                   candidate_embedding = current_canonical_embeddings[candidate_cid].to(device)
                                   similarity = F.cosine_similarity(entity_embedding.unsqueeze(0), candidate_embedding.unsqueeze(0)).item()
                                   if debug: print(f"      Comparing embedding with '{candidate_cid}'. Similarity: {similarity:.4f}")

                                   if similarity > highest_embedding_similarity:
                                        highest_embedding_similarity = similarity
                                        best_embedding_match_cid = candidate_cid # Keep track of the best one found via embedding
                              elif debug:
                                   print(f"      Skipping embedding comparison for '{candidate_cid}' - embedding not found.")

                         # Decision based on embedding results
                         if highest_embedding_similarity >= embedding_threshold:
                              best_candidate_cid = best_embedding_match_cid # Assign the one that passed embedding threshold
                              if debug: print(f"    Decision: Embedding match found! Assigning '{best_candidate_cid}' (Sim: {highest_embedding_similarity:.4f})")
                         elif debug:
                               print(f"    Decision: No embedding similarity >= {embedding_threshold} (Max was {highest_embedding_similarity:.4f}). Treating as new entity.")
                               # Will fall through to create new entity below

                    # If a best candidate was determined (either by lexical only or embedding confirmed)
                    if best_candidate_cid:
                         assigned_canonical_id = best_candidate_cid
                         # Add alias to map if it's not already there
                         if entity_text_cleaned not in current_normalization_map:
                              current_normalization_map[entity_text_cleaned] = assigned_canonical_id
                              if debug: print(f"      Added alias '{entity_text_cleaned}' -> '{assigned_canonical_id}' to map.")


            # --- Step 4: Create New Entity if No Match Found ---
            if assigned_canonical_id is None:
                 # Create new canonical ID
                 new_canonical_id = f"{entity_label}:{entity_copy['text']}" # Use original case for ID name part
                 assigned_canonical_id = new_canonical_id
                 if debug: print(f"  Decision: No match found. Creating new canonical ID: '{assigned_canonical_id}'")

                 # Add to maps
                 if entity_text_cleaned not in current_normalization_map:
                      current_normalization_map[entity_text_cleaned] = new_canonical_id
                      if debug: print(f"      Added alias '{entity_text_cleaned}' -> '{assigned_canonical_id}' to map.")
                 if entity_embedding is not None and new_canonical_id not in current_canonical_embeddings:
                      current_canonical_embeddings[new_canonical_id] = entity_embedding.cpu() # Store on CPU
                      if debug: print(f"      Added embedding for '{assigned_canonical_id}'.")
                 # Update the reverse map used for lexical search
                 if entity_text_cleaned not in canonical_name_map:
                     canonical_name_map[entity_text_cleaned] = new_canonical_id


            # Assign the final ID to the entity
            entity_copy['canonical_id'] = assigned_canonical_id
            processed_text_entities.append(entity_copy)

        updated_ner_results.append(processed_text_entities)

    return updated_ner_results, current_normalization_map, current_canonical_embeddings


# --- Example Usage (Revised) ---
if __name__ == "__main__":
    # --- Model Loading and NER ---
    model_name = "urchade/gliner_small-v2.1"
    try:
        gliner_model = GLiNER.from_pretrained(model_name)
        print(f"\nLoaded model: {model_name}")
        print(f"Model span mode: {gliner_model.config.span_mode}")

        # More diverse examples
        texts = texts = [
            # Basic Tech & Business
            "John Doe visited Paris last week. He works at Acme Corp.",
            "Apple Inc. is planning to release a new iPhone in September.",
            "Tesla's CEO, Elon Musk, announced record profits for Tesla Motors.",
            "International Business Machines (IBM) acquired Red Hat.",
            "Musk also leads SpaceX.",
            "Did Big Blue announce layoffs?",
            "The tech giant Apple is based in Cupertino.",
            "Mr. Musk tweeted about Dogecoin again.",
            "Shares of TESLA surged today after the earnings call.",
            "Microsoft reported strong growth in its Azure cloud division.",
            "Satya Nadella discussed the future of AI at the MS Build conference.",
            "Google's parent company, Alphabet Inc., faces antitrust scrutiny in Europe.",
            "Amazon Web Services (AWS) remains the leader in cloud computing.",
            "Jeff Bezos founded Amazon in Seattle.",
            "Meta Platforms, formerly Facebook, is investing heavily in the metaverse.",
            "Mark Zuckerberg testified before the U.S. Congress on Wednesday.",
            "Samsung unveiled its latest Galaxy smartphone, the S24 Ultra, in Seoul.",

            # Finance & Economics
            "The Federal Reserve (Fed) might raise interest rates next quarter.",
            "Wall Street reacted positively to the jobs report released on Friday.",
            "Berkshire Hathaway, led by Warren Buffett, reported its quarterly earnings.",
            "The European Central Bank (ECB) maintained its monetary policy.",
            "Bitcoin (BTC) and Ethereum (ETH) experienced high volatility.",
            "Goldman Sachs advised on the recent merger.",
            "The NYSE opened higher this morning.",
            "Inflation remains a key concern for the Bank of England (BoE).",
            "The deal is valued at approximately $1.5 billion USD.",

            # Politics & Government
            "President Biden addressed the nation from the White House.",
            "The United Nations Security Council convened an emergency meeting in New York.",
            "UK Prime Minister Rishi Sunak met with French President Emmanuel Macron in London.",
            "The German Chancellor, Olaf Scholz, is visiting Washington D.C.",
            "Parliament passed the new environmental bill yesterday.",
            "The Ministry of Defence confirmed the deployment.",
            "NATO summit will be held in Brussels next July.",

            # Science & Health
            "Dr. Anthony Fauci became a prominent figure during the COVID-19 pandemic.",
            "NASA's James Webb Space Telescope (JWST) delivered stunning images of the cosmos.",
            "The World Health Organization (WHO) issued new guidelines on air pollution.",
            "Researchers at MIT published a study on quantum computing.",
            "Pfizer and Moderna developed mRNA vaccines.",
            "The CDC recommends vaccination for all eligible individuals.",
            "Einstein's theory of relativity revolutionized physics.",

            # Locations & Events
            "The 2024 Summer Olympics will be held in Paris, France.",
            "Mount Everest is the highest peak on Earth.",
            "The conference took place at the Moscone Center in San Francisco (SF).",
            "He traveled from Los Angeles (LAX) to Tokyo (NRT).",
            "World War II ended in 1945.",
            "The Golden Gate Bridge is an iconic landmark.",
            "Hurricane Ian caused significant damage in Florida last year.",

            # Arts & Entertainment
            "Taylor Swift announced her Eras Tour dates for Europe.",
            "The movie 'Oppenheimer', directed by Christopher Nolan, won several Oscars.",
            "Disney Plus added new content from Marvel Studios.",
            "Leonardo da Vinci painted the Mona Lisa, housed in the Louvre Museum.",
            "The Beatles originated in Liverpool, England.",

            # Aliases/Edge Cases
            "Can Amazon deliver this by tomorrow?", # Amazon the company vs rainforest? Context is key.
            "He works for 'The Company'.", # Ambiguous org name
            "Metropolis is guarded by Superman.", # Fictional location/person
            "The report cited data from I.B.M.", # Punctuation variation
        ]
        # Expanded labels to cover examples
        labels = ["person", "location", "date", "organization", "product", "company", "event", "misc", "ceo", "ticker"] # Added ticker

        print("\nRunning NER with Embeddings...")
        ner_results_with_embeddings = get_ner_with_embeddings(gliner_model, texts, labels, threshold=0.5)
        print("NER finished.")

        # --- Normalization Setup ---
        # Initial map - Keys will be cleaned internally now
        initial_normalization_map = {
    
        }
        # Clean the keys before passing
        initial_normalization_map_cleaned = {clean_text_for_map(k): v for k, v in initial_normalization_map.items()}


        initial_canonical_embeddings = {} # Start empty, will be populated

        # --- Run Normalization ---
        print("\nRunning Normalization (Lexical First)...")
        calculation_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        final_results, final_norm_map, final_embeddings = normalize_entities_lexical_first(
            ner_results_with_embeddings,
            initial_normalization_map_cleaned, # Pass the cleaned initial map
            initial_canonical_embeddings,
            lexical_threshold=88.0,  # Adjust fuzzy threshold (0-100)
            embedding_threshold=0.90, # Similarity threshold (Cosine)
            use_embedding_confirmation=True, # Require embedding check even for single lexical match
            device=calculation_device,
            debug=True # Enable verbose output for diagnostics
        )
        print("Normalization finished.")

        # --- Display Results ---
        print("\n--- Final Normalized Results ---")
        for i, text_result in enumerate(final_results):
            print(f"\nText {i+1}: '{texts[i]}'")
            if not text_result:
                print("  No entities found.")
                continue
            for entity in text_result:
                print(f"  Entity: '{entity['text']}'")
                print(f"    Label: {entity['label']}")
                # print(f"    Score: {entity['score']:.4f}")
                # print(f"    Span: ({entity['start']}, {entity['end']})")
                print(f"    Canonical ID: {entity.get('canonical_id', 'N/A')}")
                # embedding_info = "Available" if entity.get('embedding') is not None else "Not Available"
                # print(f"    Embedding: {embedding_info}")


        print("\n--- Final Normalization Map ---")
        for alias, can_id in sorted(final_norm_map.items()):
            print(f"  '{alias}': '{can_id}'")

        print("\n--- Final Canonical Embeddings Map ---")
        print(f"  Total canonical entities with embeddings: {len(final_embeddings)}")
        # for can_id in sorted(final_embeddings.keys()):
        #     print(f"  - {can_id}")


    except ImportError as e:
        print(f"\nImport Error: {e}. Please ensure 'gliner', 'torch', 'transformers', and 'rapidfuzz' are installed.")
        print("`pip install gliner torch transformers rapidfuzz`")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure the 'gliner' library and its dependencies are installed (`pip install gliner rapidfuzz`)")
        print("Also check if the model ID is correct and accessible, and that the monkey patch was applied successfully.")
        import traceback
        traceback.print_exc()