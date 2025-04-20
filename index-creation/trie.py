import re
from collections import defaultdict

# --- 1. Word-Level Trie Implementation ---

class WordTrieNode:
    """A node in the Word-Level Trie structure."""
    def __init__(self):
        self.children = {}  # Mapping from word -> WordTrieNode
        self.is_end_of_entity = False
        # Store all original surface forms that map to this node sequence
        self.original_forms = set()
        self.count = 0
        # Store the first original form encountered for this node sequence
        self.first_encountered_form = None

class WordTrie:
    """Word-Level Trie structure for storing and tracking entity variations."""
    def __init__(self):
        self.root = WordTrieNode()
        # Store all representative forms encountered for easier lookup later
        # Map: representative_string -> list_of_words
        self._all_representatives = {}

    def _preprocess_and_tokenize(self, text):
        """Lowercase, remove common punctuation, and split into words."""
        if not isinstance(text, str): # Basic type check
            return []
        text = text.lower()
        # Remove punctuation commonly found at start/end or standalone
        # Keep internal hyphens/apostrophes. Adjust regex as needed.
        text = re.sub(r'(?:(?<=\s)|(?<=^))[,.;:"\'`!?]+|[,.;:"\'`!?]+(?:(?=\s)|(?=$))', '', text)
        # Basic removal of possessive 's which might confuse tokenization
        text = re.sub(r"\'s\b", "", text)
        # Remove remaining punctuation that's not alphanumeric, hyphen, or internal apostrophe
        text = re.sub(r'[^\w\s\'-]|(?<!\w)\'|\'(?!\w)', '', text).strip()
        words = text.split()
        # Filter out very short common words or standalone suffixes if desired (optional)
        # stop_words = {'inc', 'llc', 'ltd', 'corp', 'co'}
        # return [word for word in words if word and word not in stop_words]
        return [word for word in words if word] # Filter empty strings

    def add_entity(self, original_entity_text):
        """Adds an encountered entity based on its word sequence."""
        words = self._preprocess_and_tokenize(original_entity_text)
        if not words:
            # print(f"Skipping '{original_entity_text}' - no words after processing.") # Optional debug
            return None

        node = self.root
        for word in words:
            if word not in node.children:
                node.children[word] = WordTrieNode()
            node = node.children[word]

        # Mark the end of this specific entity sequence
        node.is_end_of_entity = True
        node.count += 1
        node.original_forms.add(original_entity_text)

        if node.first_encountered_form is None:
            node.first_encountered_form = original_entity_text
            # Store this representative form and its token sequence
            # Ensure the key exists even if words list is somehow empty (shouldn't happen with check above)
            if original_entity_text:
                 self._all_representatives[original_entity_text] = words

        return node

    def find_exact_entity_node(self, entity_text):
        """Finds the node corresponding to the exact word sequence."""
        words = self._preprocess_and_tokenize(entity_text)
        if not words:
             return None

        node = self.root
        try:
            for word in words:
                node = node.children[word]
            # Return node only if it marks the end of a known entity
            return node if node.is_end_of_entity else None
        except KeyError:
            return None # Path does not exist

    def get_entity_info(self, entity_text):
        """Retrieves tracked info for an exact entity match."""
        node = self.find_exact_entity_node(entity_text)
        if node:
            return {
                "processed_words": self._preprocess_and_tokenize(entity_text),
                "count": node.count,
                "original_forms_seen": sorted(list(node.original_forms)),
                "suggested_canonical": node.first_encountered_form
            }
        else:
            return None

    def get_all_representatives(self):
        """Returns the stored mapping of representative_form -> word_list."""
        return self._all_representatives

# --- 2. Normalization Algorithm ---

def normalize_entity_heuristic(query_text, word_trie, default_to_original=True):
    """
    Normalizes an entity using the WordTrie and a heuristic.
    Heuristic: Prefer the longest known entity that ends with the query words.
               Falls back to the exact match if no longer match is found.
    """
    query_words = word_trie._preprocess_and_tokenize(query_text)
    if not query_words:
        # Handle case where query becomes empty after processing
        # Return original text only if it was non-empty to begin with
        if query_text and default_to_original:
            return query_text
        else:
            return None


    # 1. Find exact match (if any)
    exact_match_node = word_trie.find_exact_entity_node(query_text)
    exact_match_representative = None
    if exact_match_node:
        exact_match_representative = exact_match_node.first_encountered_form

    # 2. Search for longer entities ending with the query words
    best_candidate = exact_match_representative # Start with exact match as candidate
    max_len = len(query_words) if exact_match_representative else 0

    # Get all known entities (representative form -> word list mapping)
    all_representatives = word_trie.get_all_representatives()

    # Iterate through representatives to find the best match
    # Sorting by length descending ensures we find the longest match first if multiple exist
    # (Slight optimization, but clarifies intent)
    sorted_reps = sorted(all_representatives.items(), key=lambda item: len(item[1]), reverse=True)

    for representative, candidate_words in sorted_reps:
        # Optimization: No need to check candidates shorter than or equal to current best
        if len(candidate_words) <= max_len:
             continue # Already found an equally long or longer match

        # Check if candidate ends with query words
        if candidate_words[-len(query_words):] == query_words:
             # Found a longer entity ending with the query words
             best_candidate = representative
             max_len = len(candidate_words)
             # Since we sorted by length, the first one we find is the longest
             # However, let's continue checking in case of ties in length later?
             # Let's stick with the simple logic: keep updating if strictly longer
             # (If sorting wasn't used, this loop finds *a* longest one, maybe not deterministically the first seen)

    # 3. Decide final normalization target
    if best_candidate:
        return best_candidate
    elif default_to_original and query_text: # Check query_text is not empty originally
        # No exact match, no longer match found, fallback to original
        return query_text
    else:
        return None


# --- 3. Entity Data ---
# ~100 Diverse entity examples
incoming_entities = [
    "Apple", "Apple Inc.", "apple", "APPLE", "Apple Corp.", "Apple Computer",
    "Microsoft", "microsoft", "Microsoft Corp.", "MSFT", "Microsoft Corporation",
    "Google", "google", "Google LLC", "GOOGLE",
    "Alphabet", "Alphabet Inc.", "alphabet inc",
    "Amazon", "amazon", "Amazon.com", "Amazon Web Services", "AWS",
    "Facebook", "Meta", "Meta Platforms", "facebook",
    "Tesla", "tesla", "Tesla Motors", "Tesla Inc",
    "SpaceX", "Space Exploration Technologies Corp.", "spacex",
    "NVIDIA", "Nvidia Corporation", "nvidia",
    "Intel", "Intel Corporation", "intel",
    "IBM", "International Business Machines", "I.B.M.",
    "Samsung", "Samsung Electronics", "samsung",
    "Sony", "Sony Corporation", "sony",
    "General Electric", "GE",
    "Ford", "Ford Motor Company", "ford",
    "General Motors", "GM",
    "Toyota", "Toyota Motor Corporation", "toyota",
    "Honda", "Honda Motor Co., Ltd.", "honda",
    "Coca-Cola", "Coca Cola", "Coke", "The Coca-Cola Company",
    "Pepsi", "PepsiCo", "pepsi",
    "McDonald's", "McDonalds", "mcdonalds",
    "Starbucks", "Starbucks Coffee", "starbucks",
    "Walmart", "Wal-Mart Stores, Inc.", "walmart",
    "Target", "Target Corporation", "target",
    "Costco", "Costco Wholesale Corporation", "costco",
    "Nike", "nike", "Nike, Inc.",
    "Adidas", "adidas",
    "New York City", "NYC", "N.Y.C.", "City of New York", "new york city",
    "New York", "NY", "State of New York", # Ambiguous
    "Los Angeles", "LA", "L.A.", "City of Los Angeles", "los angeles",
    "San Francisco", "SF", "S.F.", "City and County of San Francisco", "san francisco",
    "London", "london", "City of London",
    "Paris", "paris",
    "Tokyo", "tokyo",
    "United States", "USA", "U.S.A.", "United States of America", "US",
    "United Kingdom", "UK", "U.K.", "Great Britain", "GB",
    "Canada", "canada",
    "Germany", "germany",
    "France", "france",
    "China", "People's Republic of China", "china",
    "Japan", "japan",
    "Elon Musk", "Musk", "elon musk",
    "Jeff Bezos", "Bezos", "jeff bezos",
    "Bill Gates", "Gates", "bill gates",
    "Tim Cook", "Cook", "tim cook",
    "Warren Buffett", "Buffett", "warren buffett",
    "Systems", # Single word test
    "Alpha Systems",
    "Beta Systems",
    "Gamma Solutions",
    "Delta Corp",
    "Inc.", # Likely filtered out by preprocessing
    "Ltd.",
    "Corp.",
    "President Biden", "Biden", "Joe Biden",
    "Prime Minister Trudeau", "Trudeau", "Justin Trudeau",
    "Johnson & Johnson", "J&J", "johnson and johnson",
    "Hewlett-Packard", "HP", "Hewlett Packard",
    "OpenAI", "Open AI",
    "DeepMind",
    "Stanford University", "Stanford",
    "MIT", "Massachusetts Institute of Technology",
    "Harvard University", "Harvard",
    # Duplicates to test counts (though not focus of normalization output)
    "Apple", "Elon Musk", "New York City", "Systems"
]

# --- 4. Main Execution Block ---
if __name__ == "__main__":

    # 1. Initialize an empty WordTrie
    entity_tracker_word_trie = WordTrie()

    # 2. Add entities to the Trie
    print("--- Processing Incoming Entities (Word Trie) ---")
    count = 0
    for entity in incoming_entities:
        added_node = entity_tracker_word_trie.add_entity(entity)
        if added_node:
             count += 1
    print(f"Successfully added {count} entities (after processing).")
    print(f"Total unique representatives stored: {len(entity_tracker_word_trie.get_all_representatives())}")


    # 3. Normalize a selection of entities
    print("\n--- Normalization Test (Word Trie + Heuristic) ---")
    entities_to_normalize = [
        "Musk",                # Should become "Elon Musk"
        "elon musk",           # Should become "Elon Musk" (case/exact match)
        "Bezos",               # Should become "Jeff Bezos"
        "Gates",               # Should become "Bill Gates"
        "NYC",                 # Should remain "NYC" (not a suffix)
        "New York",            # Should become "New York City" (or State of New York if added)
        "City of New York",    # Should remain "City of New York" (exact match)
        "NY",                  # Should remain "NY"
        "Apple",               # Should remain "Apple" (Inc isn't suffix)
        "Apple Computer",      # Should remain "Apple Computer"
        "Inc.",                # Should be None or "Inc." depending on filtering/default
        "Systems",             # Should become "Alpha Systems" or "Beta Systems" (longer match)
        "Alpha Systems",       # Should remain "Alpha Systems"
        "US",                  # Should become "United States" (if USA/U.S.A. added first)
        "USA",                 # Should become "United States of America" (longest match)
        "J&J",                 # Should become "Johnson & Johnson"
        "HP",                  # Should become "Hewlett-Packard" or "Hewlett Packard"
        "Stanford",            # Should become "Stanford University"
        "Cook",                # Should become "Tim Cook"
        "Unknown Entity",      # Should remain "Unknown Entity" (fallback)
        "San Francisco",       # Should remain "San Francisco" (longest match ending in itself)
        "SF",                  # Should remain "SF"
        "International Business Machines", # Exact match
        "IBM",                 # Should remain "IBM"
        "Trudeau",             # Should become "Prime Minister Trudeau" or "Justin Trudeau"
        "Biden",               # Should become "President Biden" or "Joe Biden"
    ]

    for entity in entities_to_normalize:
        normalized = normalize_entity_heuristic(entity, entity_tracker_word_trie)
        # Handle None result gracefully for printing
        normalized_str = normalized if normalized is not None else "None (Filtered/Not Found)"
        print(f"Original: {entity:<35} => Normalized: {normalized_str}")

    # Optional: Print some specific entity info
    # print("\n--- Specific Entity Info ---")
    # info_musk = entity_tracker_word_trie.get_entity_info("Musk")
    # print(f"Info for 'Musk': {info_musk}")
    # info_elonmusk = entity_tracker_word_trie.get_entity_info("Elon Musk")
    # print(f"Info for 'Elon Musk': {info_elonmusk}")
    # info_systems = entity_tracker_word_trie.get_entity_info("Systems")
    # print(f"Info for 'Systems': {info_systems}")