import os
import pickle
import re
import json

def extract_number(filename):
    # Estrai il numero usando una regex
    match = re.search(r'_(\d+)\.pkl$', filename)
    return int(match.group(1)) if match else float('inf')

def process_pkl_files(input_folder):
    combined_data = []
    filtered_data = []
    count_example = 0
    correct_count = 0
    multiple_selected_count = 0
    correct_without_documents_count = 0

    output_file = os.path.join(input_folder, 'numdoc5_retr5_template_info_all.json')
    filtered_output_file = os.path.join(input_folder, 'numdoc5_retr5_template_info_all_extended.json')

    # Elenco dei file .pkl ordinati per nome
    pkl_files = sorted([filename for filename in os.listdir(input_folder) if filename.endswith('.pkl')], key=extract_number)

    print("Trovati i seguenti file .pkl:")
    for filename in pkl_files:
        print(filename)

    # Leggi tutti i file .pkl nella cartella
    for filename in pkl_files:
        file_path = os.path.join(input_folder, filename)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            combined_data.extend(data)

    for example in combined_data:
        # Controlla se almeno una risposta ha 'is_correct' = True
        is_correct = any(response.get('is_correct') for response in example.get('generated_responses', []))
        if is_correct:
            correct_count += 1

        # Controlla se ci sono più di un indice in 'selected_documents'
        selected_documents = example.get('selected_documents', [])
        if len(selected_documents) > 1:
            multiple_selected_count += 1

        # Controlla se non ci sono documenti selezionati e almeno una risposta è corretta
        if not selected_documents and is_correct:
            correct_without_documents_count += 1
            # Trova la risposta generata corretta senza documenti
            best_answer = next(
                (response.get('generated_answer') for response in example.get('generated_responses', []) if response.get('is_correct')),
                None
            )
        else:
            # Trova la risposta corretta con il punteggio BLEU più alto
            best_answer = None
            best_bleu_score = 0
            for response in example.get('generated_responses', []):
                if response.get('is_correct') and response.get('bleu_score', 0) > best_bleu_score:
                    best_bleu_score = response['bleu_score']
                    best_answer = response.get('generated_answer', None)

        filtered_data.append({
            'query': example.get('query'),
            'are_answer': is_correct,
            'generated_answer': best_answer,
            'selected_documents': selected_documents,
            'number_documents_selected': len(selected_documents)
        })

        count_example += 1

    # Salva i dati combinati in un file JSON
    with open(output_file, 'w') as f:
        json.dump(combined_data, f, indent=4)

    # Salva i dati filtrati in un file JSON
    with open(filtered_output_file, 'w') as f:
        json.dump(filtered_data, f, indent=4)

    return correct_count, multiple_selected_count, correct_without_documents_count, count_example

def main():
    """Main function to process .pkl files."""
    input_folder = r'C:\Users\franc\Documents\Bridge_the_GAP\data\gen_ids_document_training_set_bgm\nq_training\gemma-2-2b-it\train\retrieved\contriever\5_doc'
    if not os.path.isdir(input_folder):
        print("The specified folder does not exist.")
        return

    correct, multiple_selected, correct_without_documents, n_example = process_pkl_files(input_folder)

    print(f"\nNumeri di esempi totali: {n_example}")
    print(f"Esempi con risposta corretta: {correct}")
    print(f"Esempi con più di un indice in 'selected_documents': {multiple_selected}")
    print(f"Esempi senza documenti selezionati ma con risposta corretta: {correct_without_documents}")

if __name__ == "__main__":
    main()