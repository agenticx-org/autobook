import re
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict # Retained for potential internal use if extended

# Import necessary tokenizer class from transformers
try:
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
except ImportError:
    print("Error: transformers library not found.")
    print("Please install it: pip install transformers")
    exit()

# Type alias for clarity
SubwordList = List[str]
RepCacheValue = Tuple[SubwordList, str] # (subwords, entity_class)
RepCache = Dict[str, RepCacheValue] # mention -> (subwords, entity_class)

# == Subword-Level Trie (Heuristic - Class Aware) ==

class SubWordTrieNode:
    """A node in the Subword-Level Trie structure."""
    def __init__(self):
        # Mapping from subword token string to the next node
        self.children: Dict[str, SubWordTrieNode] = {}
        # Flag indicating if a complete entity sequence ends at this node
        self.is_end_of_entity: bool = False
        # Set of original mention strings that end at this node
        self.original_forms: set[str] = set()
        # Count of how many times entities ending at this node were added
        self.count: int = 0
        # The first original mention string encountered that ended at this node
        self.first_encountered_form: Optional[str] = None
        # The entity class associated with the first entity ending at this node
        self.entity_class: Optional[str] = None

class SubWordTrie:
    """
    Stores entity mentions tokenized into subwords in a Trie structure.
    Includes caching for efficient lookup during normalization.
    Requires a transformers tokenizer instance during initialization.
    """
    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        """
        Initializes the SubWordTrie.

        Args:
            tokenizer: An initialized tokenizer instance from the transformers library
                       (e.g., AutoTokenizer.from_pretrained('bert-base-uncased')).
        """
        self.root = SubWordTrieNode()
        # Cache mapping: representative_mention -> (subword_list, entity_class)
        self._all_representatives: RepCache = {}
        if tokenizer is None or not hasattr(tokenizer, 'tokenize'):
            raise ValueError("SubWordTrie requires a valid tokenizer instance with a 'tokenize' method.")
        self.tokenizer = tokenizer
        # Optional: Print tokenizer info for confirmation
        # print(f"  SubWordTrie initialized with tokenizer: {getattr(tokenizer, 'name_or_path', 'Unknown')}")

    def _preprocess_and_tokenize(self, text: str) -> SubwordList:
        """
        Performs minimal preprocessing and tokenizes text into subwords
        using the instance's tokenizer.

        Args:
            text: The input string mention.

        Returns:
            A list of subword tokens.
        """
        if not isinstance(text, str):
            return []
        text = text.strip() # Basic whitespace trim
        if not text:
            return []
        # Use the provided tokenizer's method
        subwords = self.tokenizer.tokenize(text)
        return subwords

    def add_entity(self, original_mention: str, entity_class: str) -> Optional[SubWordTrieNode]:
        """
        Adds an entity mention and its class to the Trie.

        Args:
            original_mention: The raw entity mention string.
            entity_class: The class label for the entity.

        Returns:
            The Trie node where the entity sequence ends, or None if tokenization fails.
        """
        subwords = self._preprocess_and_tokenize(original_mention)
        if not subwords:
            # print(f"Warning: Skipping '{original_mention}' - no subwords after tokenization.")
            return None # Skip if empty after tokenization

        node = self.root
        for subword in subwords:
            # Get existing child or create a new one
            node = node.children.setdefault(subword, SubWordTrieNode())

        # Mark this node as the end of an entity sequence
        node.is_end_of_entity = True
        node.count += 1
        node.original_forms.add(original_mention)

        # Store info about the first entity reaching this node
        if node.first_encountered_form is None:
            node.first_encountered_form = original_mention
            node.entity_class = entity_class
            # Also update the representative cache
            if original_mention:
                 self._all_representatives[original_mention] = (subwords, entity_class)

        return node

    def find_exact_entity_node(self, entity_text: str) -> Optional[SubWordTrieNode]:
        """
        Traverses the Trie based on the subwords of entity_text and returns
        the final node if it represents a complete, known entity.

        Args:
            entity_text: The entity mention string to search for.

        Returns:
            The SubWordTrieNode if found and is an end node, otherwise None.
        """
        subwords = self._preprocess_and_tokenize(entity_text)
        if not subwords:
            return None
        node = self.root
        try:
            for subword in subwords:
                node = node.children[subword]
            # Return node only if it marks the end of a known entity sequence
            return node if node.is_end_of_entity else None
        except KeyError:
            # Path does not exist in the Trie
            return None

    def get_all_representatives_with_class(self) -> RepCache:
        """
        Returns the cached dictionary mapping representative mentions to their
        (subword_list, entity_class) tuples.
        """
        return self._all_representatives

# --- Normalization Function (Class-Aware Heuristic) ---
def normalize_subword_trie_class_aware(
    query_mention: str,
    query_class: str, # Requires the class of the query entity
    subword_trie: SubWordTrie,
    default_to_original: bool = True
) -> Optional[str]:
    """
    Normalizes a query mention using a class-aware Subword Trie heuristic.

    The heuristic prefers the longest known entity mention that:
    1. Belongs to the *same entity class* as the query mention.
    2. Ends with the *exact same subword sequence* as the query mention.

    If multiple longest matches exist, the one encountered first during the
    (implementation-dependent) iteration might be chosen. If no longer match
    is found, it checks for an exact match (of the correct class).

    Args:
        query_mention: The entity mention string to normalize.
        query_class: The entity class label of the query mention.
        subword_trie: An initialized and populated SubWordTrie instance.
        default_to_original: If True, returns the original query_mention if
                             no suitable normalization target is found. If False,
                             returns None in that case.

    Returns:
        The normalized entity mention string, the original mention (if falling back),
        or None.
    """
    query_subwords = subword_trie._preprocess_and_tokenize(query_mention)
    # Handle empty query after tokenization
    if not query_subwords:
        return query_mention if default_to_original and query_mention else None

    # 1. Check for an exact match that also matches the query class
    exact_match_node = subword_trie.find_exact_entity_node(query_mention)
    exact_match_representative = None
    if exact_match_node and exact_match_node.entity_class == query_class:
        # Ensure the node has a representative form stored
        if exact_match_node.first_encountered_form is not None:
            exact_match_representative = exact_match_node.first_encountered_form

    # 2. Search for longer candidate entities OF THE SAME CLASS ending with the query subwords
    best_candidate = exact_match_representative # Initialize with class-aware exact match (or None)
    # Initialize max_len based on whether a valid exact match was found
    max_len = len(query_subwords) if exact_match_representative is not None else 0

    all_representatives_with_class = subword_trie.get_all_representatives_with_class()
    # Sort candidates by the length of their subword sequence (descending)
    # This helps prioritize longer matches efficiently
    sorted_reps = sorted(
        all_representatives_with_class.items(),
        key=lambda item: len(item[1][0]), # item[1] is (subwords, class), item[1][0] is subwords
        reverse=True
    )

    for representative, (candidate_subwords, candidate_class) in sorted_reps:
        # --- Class Constraint ---
        if candidate_class != query_class:
            continue # Skip if the candidate is not of the same class as the query

        # --- Length Constraint (Only consider *strictly* longer candidates) ---
        # If candidate length equals max_len, it's either the exact match we already
        # considered or another entity of the same length, neither preferred over current best.
        if len(candidate_subwords) <= max_len:
            continue # Optimization: Already found a match of this length or longer

        # --- Suffix Constraint (Check if candidate ends with query subwords) ---
        if candidate_subwords[-len(query_subwords):] == query_subwords:
            # Found a suitable longer candidate of the correct class
            best_candidate = representative
            max_len = len(candidate_subwords)
            # Optional optimization: Since sorted by length, first longer match found is the longest.
            # break

    # 3. Return the best candidate found or fallback
    if best_candidate is not None:
        return best_candidate
    elif default_to_original and query_mention:
        # Fallback to original only if no suitable candidate was found
        return query_mention
    else:
        # No candidate found and fallback is disabled or query was empty
        return None


# --- Example Usage ---
if __name__ == "__main__":
    print("--- Subword Trie (Class-Aware) Example ---")

    # --- Setup ---
    TOKENIZER_NAME = "bert-base-uncased" # Or another suitable tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        print(f"Initialized tokenizer: {TOKENIZER_NAME}")
    except Exception as e:
        print(f"Failed to initialize tokenizer '{TOKENIZER_NAME}': {e}")
        exit()

    subword_trie_instance = SubWordTrie(tokenizer=tokenizer)

    # --- Populate with sample data ---
    print("\nPopulating Trie...")
    sample_data = [
        ("Musk", "person"),
        ("Elon Musk", "person"),
        ("Tesla", "company"),
        ("SpaceX", "company"),
        ("Gates", "person"),
        ("Bill Gates", "person"),
        ("Microsoft", "company"), # Note: Treat ORG/COMPANY consistently
        ("International Business Machines", "company"),
        ("IBM", "company"),
        ("UC Berkeley", "university"),
        ("University of California, Berkeley", "university"),
        ("Apple", "company"),
        ("The Big Apple", "city"), # Different class, same ending word
        ("Fauci", "person"),
        ("Dr. Fauci", "person"),
        ("Anthony Fauci", "person"),
    ]
    for mention, e_class in sample_data:
        subword_trie_instance.add_entity(mention, e_class)
    print("Population complete.")

    # --- Test Normalization ---
    print("\nTesting Normalization...")
    test_cases = [
        ("Musk", "person"),          # Should become Elon Musk
        ("Gates", "person"),         # Should become Bill Gates
        ("IBM", "company"),          # Should remain IBM (no longer match)
        ("UC Berkeley", "university"),# Should become University of California, Berkeley
        ("Apple", "company"),        # Should remain Apple (no longer match *of same class*)
        ("Fauci", "person"),         # Should become Dr. Fauci or Anthony Fauci (longest)
        ("Unknown", "person"),       # Should remain Unknown (fallback)
        ("Berkeley", "university"),  # Should remain Berkeley (no match ending in it)
    ]

    print("-" * 60)
    print(f"{'Original Mention':<25} | {'Query Class':<15} | {'Normalized':<25}")
    print("-" * 60)
    for mention, q_class in test_cases:
        normalized = normalize_subword_trie_class_aware(
            mention, q_class, subword_trie_instance
        )
        normalized_str = normalized if normalized is not None else "None"
        print(f"{mention:<25} | {q_class:<15} | {normalized_str:<25}")
    print("-" * 60)