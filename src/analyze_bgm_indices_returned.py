import json
import re
from collections import Counter

def analyze_bgm_indices(file_path):
    """
    Analizza il file JSON della generazione dei migliori indici con il 600° step del training.
    Conta quanti esempi hanno 1,2,3,4,5 ID in bgm_indices e quanti non ne hanno trovati,
    escludendo gli ID che contengono 'Unknown()' o 'Unknown("")'.
    Inoltre, controlla se nel prompt è presente una delle risposte nel campo "answers",
    considerando solo il contenuto dopo "Documents:" e associandolo alla categoria di documenti trovati.

    Args:
        file_path (str): Percorso del file JSON.

    Returns:
        dict: Statistiche sul numero di ID in bgm_indices e presenza di risposte nel prompt.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Conta quante volte troviamo 0, 1, 2, 3, 4, 5 ID
    bgm_counts = Counter()
    answer_in_prompt_counts = Counter()

    for example in data:
        bgm_indices = example.get("bgm_indices", "")
        prompt = example.get("prompt", "").lower()
        answers = [ans.lower() for ans in example.get("answers", [])]
        
        # Estrarre solo la parte dopo "Documents:" se presente
        if "documents:" in prompt:
            prompt = prompt.split("documents:", 1)[-1]
        else:
            prompt = "" 

        answer_in_prompt = any(answer in prompt for answer in answers) if prompt else False

        # Se non ha trovato nessun ID buono
        if bgm_indices in ["NO_DOCS", "Unknown("")"]:
            bgm_counts[0] += 1
            if answer_in_prompt:
                answer_in_prompt_counts[0] += 1
        else:

            ids = [id_.strip() for id_ in bgm_indices.split(",") if id_.strip()]

            valid_ids = [id_ for id_ in ids if not re.match(r"Unknown\(.*\)", id_)]

            # Conta quanti ID validi ci sono
            num_valid_ids = len(valid_ids)

            # Se non rimangono ID validi, conta come 0 documenti
            bgm_counts[num_valid_ids] += 1
            if answer_in_prompt:
                answer_in_prompt_counts[num_valid_ids] += 1

    stats = {
        "num_examples_0_docs": bgm_counts[0],
        "num_examples_1_doc": bgm_counts[1],
        "num_examples_2_docs": bgm_counts[2],
        "num_examples_3_docs": bgm_counts[3],
        "num_examples_4_docs": bgm_counts[4],
        "num_examples_5_docs": bgm_counts[5],
        "total_examples": len(data),
        "num_examples_with_answer_in_context_0_docs": answer_in_prompt_counts[0],
        "num_examples_with_answer_in_context_1_doc": answer_in_prompt_counts[1],
        "num_examples_with_answer_in_context_2_docs": answer_in_prompt_counts[2],
        "num_examples_with_answer_in_context_3_docs": answer_in_prompt_counts[3],
        "num_examples_with_answer_in_context_4_docs": answer_in_prompt_counts[4],
        "num_examples_with_answer_in_context_5_docs": answer_in_prompt_counts[5],
    }

    return stats

if __name__ == "__main__":

    file_path = r"C:\Users\franc\Documents\Bridge_the_GAP\data\gen_best_ids_doc_test_set_according_bgm\nq\gemma-2-2b-it\test\retrieved\contriever\5_doc\numdoc5_retr5_template_bgm_training_steps_1700_info.json"

    stats = analyze_bgm_indices(file_path)

    print("Statistiche sugli ID in bgm_indices:")
    for key, value in stats.items():
        print(f"{key}: {value}")